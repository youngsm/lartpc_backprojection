import numpy as np
import torch
import math
import time

def backproject_hits_into_lines_cuda(projection_data, theta, u_min, volume_shape, device="cuda", plane_id=None):
    """
    Backproject 2D hits into 3D lines using CUDA.
    
    Args:
        projection_data (torch.Tensor): 2D projection data (x, u)
        theta (float): Angle of the wire plane in radians
        u_min (float): Minimum u-coordinate
        volume_shape (tuple): Shape of the 3D volume (N, N, N)
        device (str): Device to use ('cuda' or 'cpu')
        plane_id (int, optional): The ID of the wire plane. If None, will be set to 0.
        
    Returns:
        tuple: (points, directions, plane_id) representing the backprojected lines
    """
    # Get non-zero hits from the projection
    hits = torch.nonzero(projection_data > 0, as_tuple=True)
    x_indices = hits[0]
    u_indices = hits[1]
    
    # Create line parameters
    sin_theta = math.sin(theta)
    cos_theta = math.cos(theta)
    
    # Number of hits
    n_hits = x_indices.size(0)
    
    # For each hit, compute a point on the line
    points = torch.zeros((n_hits, 3), device=device)
    points[:, 0] = x_indices.float()  # x-coordinate is preserved
    
    # Compute u-coordinate in the original space
    u_orig = u_indices.float() + u_min
    
    # For the vertical case (theta = 0), u = z
    if abs(sin_theta) < 1e-10:
        points[:, 1] = 0.0  # y = 0
        points[:, 2] = u_orig  # z = u
    # For the horizontal case (theta = 90 degrees), u = -y
    elif abs(cos_theta) < 1e-10:
        points[:, 1] = -u_orig  # y = -u
        points[:, 2] = 0.0  # z = 0
    # For the general case
    else:
        # Choose a point where either y=0 or z=0
        points[:, 1] = 0.0  # y = 0
        points[:, 2] = u_orig / cos_theta  # z = u / cos(theta)
    
    # Compute direction vectors (perpendicular to the wire direction)
    directions = torch.zeros((n_hits, 3), device=device)
    directions[:, 0] = 0.0  # Perpendicular to x (the drift direction)
    directions[:, 1] = cos_theta  # y-component
    directions[:, 2] = sin_theta  # z-component
    
    # Normalize directions
    norm = torch.sqrt(torch.sum(directions ** 2, dim=1, keepdim=True))
    directions = directions / norm
    
    # Set plane ID (use provided plane_id or default to 0)
    if plane_id is None:
        plane_id = 0
    plane_ids = torch.ones(n_hits, device=device, dtype=torch.int32) * plane_id
    
    return points, directions, plane_ids

def project_coordinates_to_plane(coords, values, volume_shape, theta, u_min, device='cuda', projection_size=None):
    """
    Unified function to project a sparse 3D volume to a 2D plane with optional diffusion effects.
    Supports both standard and differentiable projection modes.
    
    Args:
        coords (torch.Tensor): Coordinates of non-zero voxels (N, 3)
        values (torch.Tensor): Values of non-zero voxels (N)
        volume_shape (tuple): Shape of the volume (x_size, y_size, z_size)
        theta (float): Projection angle in radians
        u_min (float): Minimum u-coordinate
        device (str): Device to use for computation
        projection_size (tuple, optional): Size of the projection (x_size, u_size)
        
    Returns:
        torch.Tensor: Projection of shape (x_size, u_size)
    """
    # Edge case: empty volume
    if coords.shape[0] == 0:
        if projection_size is not None:
            return torch.zeros(projection_size, device=device)
        return torch.zeros((volume_shape[0], 1), device=device)
    
    # Sort coordinates for better memory access patterns
    # Sort by x coordinate first, which will improve cache locality
    sorted_indices = torch.argsort(coords[:, 0])
    sorted_coords = coords[sorted_indices]
    sorted_values = values[sorted_indices]
    
    # Determine projection dimensions
    x_size = volume_shape[0]
    
    if projection_size is None:
        # Automatically determine projection size
        # Maximum u can be calculated from the volume dimensions and angle
        max_y = volume_shape[1] - 1
        max_z = volume_shape[2] - 1
        
        # Calculate max u value (account for projection angle)
        sin_theta = torch.sin(torch.tensor(theta, device=device))
        cos_theta = torch.cos(torch.tensor(theta, device=device))
        u_max = -max_y * sin_theta + max_z * cos_theta
        
        # Projection size is (x_size, u_size)
        u_size = int(torch.ceil(u_max - u_min).item()) + 1
        projection_size = (x_size, u_size)
    else:
        x_size, u_size = projection_size
    
    # Get needed trig values
    sin_theta = torch.sin(torch.tensor(theta, device=device))
    cos_theta = torch.cos(torch.tensor(theta, device=device))
    
    # Calculate u coordinates for each point
    u_coords = -sorted_coords[:, 1] * sin_theta + sorted_coords[:, 2] * cos_theta - u_min

    # For differentiable mode, use bilinear interpolation to maintain gradient flow
    # Use floor and ceil to properly distribute contribution
    u_lower = torch.floor(u_coords).long()
    u_upper = u_lower + 1
    
    # Calculate weights for bilinear interpolation
    u_weight_upper = u_coords - u_lower.float()
    u_weight_lower = 1.0 - u_weight_upper
    
    # Get x indices
    x_indices = sorted_coords[:, 0].long()
    
    # Filter out indices that would be outside the projection
    # For lower u indices
    valid_mask_lower = (x_indices >= 0) & (x_indices < x_size) & \
                    (u_lower >= 0) & (u_lower < u_size)
    
    # For upper u indices
    valid_mask_upper = (x_indices >= 0) & (x_indices < x_size) & \
                    (u_upper >= 0) & (u_upper < u_size)
    
    # Create sparse tensors for each contribution
    # Lower u contribution
    if valid_mask_lower.any():
        indices_lower = torch.stack([x_indices[valid_mask_lower], u_lower[valid_mask_lower]], dim=0)
        values_lower = (
            sorted_values[valid_mask_lower] * u_weight_lower[valid_mask_lower]
        )
        lower_contribution = torch.sparse_coo_tensor(
            indices_lower, 
            values_lower,
            size=(x_size, u_size),
            device=device
        )
    else:
        lower_contribution = torch.sparse_coo_tensor(
            torch.zeros((2, 0), dtype=torch.long, device=device),
            torch.zeros(0, device=device),
            size=(x_size, u_size),
            device=device
        )
    
    # Upper u contribution
    if valid_mask_upper.any():
        indices_upper = torch.stack([x_indices[valid_mask_upper], u_upper[valid_mask_upper]], dim=0)
        values_upper = (
            sorted_values[valid_mask_upper] * u_weight_upper[valid_mask_upper]
        )
        upper_contribution = torch.sparse_coo_tensor(
            indices_upper, 
            values_upper,
            size=(x_size, u_size),
            device=device
        )
    else:
        upper_contribution = torch.sparse_coo_tensor(
            torch.zeros((2, 0), dtype=torch.long, device=device),
            torch.zeros(0, device=device),
            size=(x_size, u_size),
            device=device
        )
    
    # Sum the contributions and convert to dense tensor
    projection = (lower_contribution + upper_contribution).to_dense()    
    return projection

def project_coordinates_to_plane_exact(
    coords, values, volume_shape, theta, u_min, device="cuda", projection_size=None
):
    """
    Project a sparse 3D volume to a 2D plane assuming exact mapping to integer U coordinates.
    This version removes bilinear interpolation for better performance when coordinates align with grid.

    Args:
        coords (torch.Tensor): Coordinates of non-zero voxels (N, 3)
        values (torch.Tensor): Values of non-zero voxels (N)
        volume_shape (tuple): Shape of the volume (x_size, y_size, z_size)
        theta (float): Projection angle in radians
        u_min (float): Minimum u-coordinate
        device (str): Device to use for computation
        projection_size (tuple, optional): Size of the projection (x_size, u_size)

    Returns:
        torch.Tensor: Projection of shape (x_size, u_size)
    """
    # Edge case: empty volume
    if coords.shape[0] == 0:
        if projection_size is not None:
            return torch.zeros(projection_size, device=device)
        return torch.zeros((volume_shape[0], 1), device=device)

    # Sort coordinates for better memory access patterns
    sorted_indices = torch.argsort(coords[:, 0])
    sorted_coords = coords[sorted_indices]
    sorted_values = values[sorted_indices]

    # Determine projection dimensions
    x_size = volume_shape[0]

    if projection_size is None:
        # Automatically determine projection size
        max_y = volume_shape[1] - 1
        max_z = volume_shape[2] - 1

        sin_theta = torch.sin(torch.tensor(theta, device=device))
        cos_theta = torch.cos(torch.tensor(theta, device=device))
        u_max = -max_y * sin_theta + max_z * cos_theta

        u_size = int(torch.ceil(u_max - u_min).item()) + 1
        projection_size = (x_size, u_size)
    else:
        x_size, u_size = projection_size

    # Get needed trig values
    sin_theta = torch.sin(torch.tensor(theta, device=device))
    cos_theta = torch.cos(torch.tensor(theta, device=device))

    # Calculate u coordinates for each point and round to nearest integer
    u_coords = (
        -sorted_coords[:, 1] * sin_theta + sorted_coords[:, 2] * cos_theta - u_min
    )
    u_indices = torch.round(u_coords).long()

    # Get x indices
    x_indices = sorted_coords[:, 0].long()

    # Filter out indices that would be outside the projection
    valid_mask = (
        (x_indices >= 0)
        & (x_indices < x_size)
        & (u_indices >= 0)
        & (u_indices < u_size)
    )

    # Create sparse tensor for the projection
    if valid_mask.any():
        indices = torch.stack([x_indices[valid_mask], u_indices[valid_mask]], dim=0)
        projection_values = sorted_values[valid_mask]

        # Handle duplicate indices by summing the values
        projection_sparse = torch.sparse_coo_tensor(
            indices, projection_values, size=(x_size, u_size), device=device
        ).coalesce()  # Coalesce sums duplicate indices

        projection = projection_sparse.to_dense()
    else:
        projection = torch.zeros(projection_size, device=device)

    return projection

def find_intersections_between_lines_cuda(
    points1,
    directions1,
    plane_ids1,
    points2,
    directions2,
    plane_ids2,
    tolerance=1e-2,
    device="cuda",
    debug=False,
    snap_to_grid=True,
    voxel_size=1.0,
):
    """
    Find intersections between two sets of lines using direct solving.

    This is fine when the lines are in the same x-plane, but not when they are not.
    In this project, they are. But in real life, they are certainly not.

    ---

    Given two sets of lines represented by points p_1, p_2 and direction vectors d_1, d_2,
    we find their intersection by solving:

    p_1 + t_1 * d_1 = p_2 + t_2 * d_2

    For lines in the same x-plane, this becomes a 2D problem in the y-z plane,
    represented as a linear system:

    [  d_1y  -d_2y ] [ t_1 ] = [ p_2y - p_1y ]
    [  d_1z  -d_2z ] [ t_2 ] = [ p_2z - p_1z ]

    Using Cramer's rule, the solution is:

                      | p_2y-p_1y  -d_2y |                     | d_1y  p_2y-p_1y |
    t_1 = det(A)^-1 * | p_2z-p_1z  -d_2z |,  t_2 = det(A)^-1 * | d_1z  p_2z-p_1z |

    where det(A) = d_1y * d_2z - d_2y * d_1z

    Once t_1 and t_2 are found, the intersection point is:
    p_intersect = p_1 + t_1 * d_1 = p_2 + t_2 * d_2

    If the lines truly intersect, then the above equation is true. If they don't actually
    intersect, then p_1 + t_1 * d_1 will give the point on line 1 of closest approach to line 2,
    and p_2 + t_2 * d_2 will give the point on line 2 of closest approach to line 1. In this case,
    we'd just take the average of the two points as the intersection point.

    Args:
        points1 (torch.Tensor): Points on the first set of lines (N, 3)
        directions1 (torch.Tensor): Direction vectors of the first set of lines (N, 3)
        plane_ids1 (torch.Tensor): Plane IDs for the first set of lines (N)
        points2 (torch.Tensor): Points on the second set of lines (M, 3)
        directions2 (torch.Tensor): Direction vectors of the second set of lines (M, 3)
        plane_ids2 (torch.Tensor): Plane IDs for the second set of lines (M)
        tolerance (float): Tolerance for testing whether two lines are in the same x-plane
        device (str): Device to use ('cuda' or 'cpu')
        debug (bool): Whether to print debug information
        snap_to_grid (bool): Whether to snap intersection points to the nearest grid points
        voxel_size (float): Size of the voxels in the volume
    Returns:
        tuple: (intersection_points, line_indices1, line_indices2, distances)
    """
    debug = True
    if debug:
        start_time = time.time()
        print(
            f"Starting intersection calculation between {points1.shape[0]} and {points2.shape[0]} lines..."
        )
        print(f"Checking if direct solving is possible...")

    # optional check if all lines are pointed in y-z planes (which they should!)
    # all_x_planar1 = torch.all(torch.abs(directions1[:, 0]) < 1e-10).item()
    # all_x_planar2 = torch.all(torch.abs(directions2[:, 0]) < 1e-10).item()
    # assert all_x_planar1 and all_x_planar2, "Lines are not in y-z planes"

    # Get sizes
    n = points1.size(0)
    m = points2.size(0)

    # Expand tensors for broadcasting
    p1 = points1.unsqueeze(1).expand(n, m, 3)  # (n, m, 3)
    d1 = directions1.unsqueeze(1).expand(n, m, 3)  # (n, m, 3)
    p2 = points2.unsqueeze(0).expand(n, m, 3)  # (n, m, 3)
    d2 = directions2.unsqueeze(0).expand(n, m, 3)  # (n, m, 3)

    # Create plane mask - must be from different planes
    plane_mask = plane_ids1.unsqueeze(1) != plane_ids2.unsqueeze(0)

    # Check x-coordinate similarity between line origins
    x_diff = torch.abs(p1[..., 0] - p2[..., 0])
    same_x_plane = x_diff < tolerance

    assert same_x_plane.eq(p1[...,0].eq(p2[...,0])).all(), "Lines are not in the same x-plane"

    # Lines that can be directly solved: different planes, same x-plane
    direct_solvable = plane_mask & same_x_plane

    if debug:
        num_solvable = torch.sum(direct_solvable).item()
        print(f"Found {num_solvable} directly solvable line pairs out of {n * m} total pairs")

    # Setup system of equations for directly solvable pairs
    # For lines in y-z plane, we need to solve:
    # p1_y + t1*d1_y = p2_y + t2*d2_y
    # p1_z + t1*d1_z = p2_z + t2*d2_z

    # Extract the relevant line pairs
    solvable_indices = torch.nonzero(direct_solvable, as_tuple=True)
    idx1, idx2 = solvable_indices

    # Get the y-z components for the solvable pairs
    p1_yz = p1[direct_solvable][:, 1:]  # (num_solvable, 2) - y,z components
    d1_yz = d1[direct_solvable][:, 1:]
    p2_yz = p2[direct_solvable][:, 1:]
    d2_yz = d2[direct_solvable][:, 1:]

    # Setup system: [d1_y, d2_y; d1_z, d2_z] * [t1; t2] = [p2_y - p1_y; p2_z - p1_z]
    a11 = d1_yz[:, 0]  # d1_y
    a12 = d2_yz[:, 0]  # d2_y
    a21 = d1_yz[:, 1]  # d1_z
    a22 = d2_yz[:, 1]  # d2_z

    b1 = p2_yz[:, 0] - p1_yz[:, 0]  # p2_y - p1_y
    b2 = p2_yz[:, 1] - p1_yz[:, 1]  # p2_z - p1_z

    # determinant
    det = a11 * a22 - a12 * a21
    # Check for parallel lines (det near zero)
    valid_det = torch.abs(det) > 1e-10

    # solve for t1 and t2 where det is valid
    t1 = torch.zeros_like(det)
    t2 = torch.zeros_like(det)

    t1[valid_det] = (a22[valid_det] * b1[valid_det] - a12[valid_det] * b2[valid_det]) / det[valid_det]
    t2[valid_det] = (a21[valid_det] * b1[valid_det] - a11[valid_det] * b2[valid_det]) / det[valid_det]

    # p_intersect = p1 + t1 * d1
    intersect_points = torch.zeros((torch.sum(valid_det).item(), 3), device=device)

    # Valid indices among the solvable pairs
    valid_idx1 = idx1[valid_det]
    valid_idx2 = idx2[valid_det]
    valid_t1 = t1[valid_det]
    
    # Calculate intersection points
    intersect_points[:, 0] = points1[valid_idx1, 0]  # Use x from first line
    intersect_points[:, 1] = points1[valid_idx1, 1] + valid_t1 * directions1[valid_idx1, 1]
    intersect_points[:, 2] = points1[valid_idx1, 2] + valid_t1 * directions1[valid_idx1, 2]
    
    # Calculate distances between calculated points on both lines
    # just for consistency with the closest-points API
    p1_intersect = intersect_points
    p2_intersect = torch.zeros_like(intersect_points)
    
    # Points on line 2
    valid_t2 = t2[valid_det]
    p2_intersect[:, 0] = points2[valid_idx2, 0]
    p2_intersect[:, 1] = points2[valid_idx2, 1] + valid_t2 * directions2[valid_idx2, 1]
    p2_intersect[:, 2] = points2[valid_idx2, 2] + valid_t2 * directions2[valid_idx2, 2]

    # take average between the two intersection points (also points closest to the other line)
    # if the lines actually intersect, then both p1_intersect and p2_intersect should be the same point
    # and this average is redundant. but irl we don't have perfect times for the intersections,
    # so X-values will be different.
    intersect_points = (p1_intersect + p2_intersect) / 2.0

    # Snap to grid if requested
    if snap_to_grid:
        intersect_points = (
            torch.round(intersect_points / voxel_size)
        )

    if debug:
        end_time = time.time()
        print(f"Direct solving found {intersect_points.shape[0]} intersections")
        print(f"Total processing time: {end_time - start_time:.2f} seconds")

    return intersect_points, idx1, valid_idx2