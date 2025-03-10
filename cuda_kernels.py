import numpy as np
import torch
import math
import time


def get_line_parameters_tensor(lines, device="cuda"):
    """
    Convert a list of Line3D objects to tensors of points and directions for CUDA processing.

    Args:
        lines (list): List of Line3D objects
        device (str): Device to store tensors on ('cuda' or 'cpu')

    Returns:
        tuple: (points_tensor, directions_tensor, plane_ids_tensor)
    """
    points = np.array([line.point for line in lines], dtype=np.float32)
    directions = np.array([line.direction for line in lines], dtype=np.float32)
    plane_ids = np.array(
        [line.plane_id if line.plane_id is not None else -1 for line in lines],
        dtype=np.int32,
    )

    points_tensor = torch.tensor(points, device=device)
    directions_tensor = torch.tensor(directions, device=device)
    plane_ids_tensor = torch.tensor(plane_ids, device=device)

    return points_tensor, directions_tensor, plane_ids_tensor


def closest_points_between_lines_cuda(points1, directions1, points2, directions2):
    """
    Compute the closest points between two sets of lines using CUDA.

    Args:
        points1 (torch.Tensor): Points on the first set of lines (N, 3)
        directions1 (torch.Tensor): Direction vectors of the first set of lines (N, 3)
        points2 (torch.Tensor): Points on the second set of lines (M, 3)
        directions2 (torch.Tensor): Direction vectors of the second set of lines (M, 3)

    Returns:
        tuple: (closest_points1, closest_points2, distances) - Closest points on each line and the distances between them
    """
    # Ensure inputs are on CUDA
    device = points1.device

    # Get sizes
    n = points1.size(0)
    m = points2.size(0)

    # Compute closest points for all pairs
    # This is done by solving a 2x2 linear system for each pair

    # Expand tensors for broadcasting
    p1 = points1.unsqueeze(1).expand(n, m, 3)  # (n, m, 3)
    d1 = directions1.unsqueeze(1).expand(n, m, 3)  # (n, m, 3)
    p2 = points2.unsqueeze(0).expand(n, m, 3)  # (n, m, 3)
    d2 = directions2.unsqueeze(0).expand(n, m, 3)  # (n, m, 3)

    # Compute coefficients of the 2x2 linear system
    # [a, b; c, d] * [t1; t2] = [e; f]

    a = torch.sum(d1 * d1, dim=2)  # (n, m)
    b = -torch.sum(d1 * d2, dim=2)  # (n, m)
    c = b  # (n, m)
    d = torch.sum(d2 * d2, dim=2)  # (n, m)

    dp = p2 - p1  # (n, m, 3)
    e = torch.sum(dp * d1, dim=2)  # (n, m)
    f = -torch.sum(dp * d2, dim=2)  # (n, m)

    # Compute determinant
    det = a * d - b * c  # (n, m)

    # Handle parallel lines (det close to 0)
    parallel_mask = torch.abs(det) < 1e-10

    # For non-parallel lines, solve the system
    # For parallel lines, set parameters to 0
    t1 = torch.zeros_like(det)
    t2 = torch.zeros_like(det)

    # Solve for t1 and t2 where det != 0
    non_parallel_mask = ~parallel_mask
    t1[non_parallel_mask] = (
        d[non_parallel_mask] * e[non_parallel_mask]
        - b[non_parallel_mask] * f[non_parallel_mask]
    ) / det[non_parallel_mask]
    t2[non_parallel_mask] = (
        a[non_parallel_mask] * f[non_parallel_mask]
        - c[non_parallel_mask] * e[non_parallel_mask]
    ) / det[non_parallel_mask]

    # Compute closest points
    # p1 + t1 * d1 and p2 + t2 * d2
    t1 = t1.unsqueeze(2).expand(n, m, 3)  # (n, m, 3)
    t2 = t2.unsqueeze(2).expand(n, m, 3)  # (n, m, 3)

    closest1 = p1 + t1 * d1  # (n, m, 3)
    closest2 = p2 + t2 * d2  # (n, m, 3)

    # Compute distances between closest points
    distances = torch.sqrt(torch.sum((closest1 - closest2) ** 2, dim=2))  # (n, m)

    return closest1, closest2, distances


def find_intersections_cuda(
    points1,
    directions1,
    plane_ids1,
    points2,
    directions2,
    plane_ids2,
    tolerance=1.0,
    device="cuda",
    debug=False,
    snap_to_grid=True
):
    """
    Find intersections between two sets of lines using CUDA.
    
    Args:
        points1 (torch.Tensor): Points on the first set of lines (N, 3)
        directions1 (torch.Tensor): Direction vectors of the first set of lines (N, 3)
        plane_ids1 (torch.Tensor): Plane IDs for the first set of lines (N)
        points2 (torch.Tensor): Points on the second set of lines (M, 3)
        directions2 (torch.Tensor): Direction vectors of the second set of lines (M, 3)
        plane_ids2 (torch.Tensor): Plane IDs for the second set of lines (M)
        tolerance (float): Tolerance for intersection testing
        device (str): Device to use ('cuda' or 'cpu')
        debug (bool): Whether to print debug information
        snap_to_grid (bool): Whether to snap intersection points to the nearest grid points
        
    Returns:
        tuple: (intersection_points, line_indices1, line_indices2, distances)
    """
    if debug:
        start_time = time.time()
        print(f"Starting intersection calculation between {points1.shape[0]} and {points2.shape[0]} lines...")

    # Compute closest points between all pairs of lines
    closest1, closest2, distances = closest_points_between_lines_cuda(points1, directions1, points2, directions2)
    
    if debug:
        closest_time = time.time()
        print(f"  Computed closest points in {closest_time - start_time:.2f} seconds")

    # Create masks for valid intersections
    # 1. Distance is below tolerance
    distance_mask = distances <= tolerance
    
    # 2. Lines are from different planes
    plane_mask = plane_ids1.unsqueeze(1) != plane_ids2.unsqueeze(0)
    
    # 3. Combined mask
    valid_mask = distance_mask & plane_mask
    
    # Count valid intersections
    num_intersections = torch.sum(valid_mask).item()
    
    if debug:
        mask_time = time.time()
        print(f"  Created intersection masks in {mask_time - closest_time:.2f} seconds")
        print(f"  Found {num_intersections} valid intersections out of {points1.shape[0] * points2.shape[0]} potential pairs")
    
    # Get indices of valid intersections
    indices = torch.nonzero(valid_mask, as_tuple=True)
    line_indices1 = indices[0]
    line_indices2 = indices[1]
    
    # Compute intersection points as midpoints of closest points
    intersection_points = (closest1[valid_mask] + closest2[valid_mask]) / 2.0
    
    # Snap intersection points to the nearest grid points if requested
    if snap_to_grid:
        intersection_points = torch.round(intersection_points)
    
    # Get distances of valid intersections
    valid_distances = distances[valid_mask]
    
    if debug:
        end_time = time.time()
        print(f"  Extracted valid intersections in {end_time - mask_time:.2f} seconds")
        if snap_to_grid:
            print(f"  Snapped intersection points to nearest grid points")
        print(f"  Total intersection processing time: {end_time - start_time:.2f} seconds")
    
    return intersection_points, line_indices1, line_indices2, valid_distances


def merge_nearby_intersections_cuda(intersection_points, distances, tolerance=1.0, batch_size=10000, debug=False):
    """
    Merge intersection points that are very close to each other, using a memory-efficient approach.
    
    Args:
        intersection_points (torch.Tensor): Intersection points (N, 3)
        distances (torch.Tensor): Distances between the lines (N)
        tolerance (float): Tolerance for merging
        batch_size (int): Size of batches for processing to limit memory usage
        debug (bool): Whether to print debug information
        fast_mode (bool): Whether to use the fastest implementation (less precise but much faster)
        
    Returns:
        torch.Tensor: Merged intersection points
    """
    if debug:
        start_time = time.time()
        print(f"Starting point merging for {intersection_points.shape[0]} points (tolerance={tolerance})...")
    
    # If no intersections, return empty tensor
    if intersection_points.size(0) == 0:
        if debug:
            print("  No intersection points to merge.")
        return torch.zeros((0, 3), device=intersection_points.device)
    
    result = _merge_nearby_intersections(intersection_points, distances, tolerance, debug)
    
    if debug:
        end_time = time.time()
        print(f"  Merged {intersection_points.shape[0]} points into {result.shape[0]} clusters")
        print(f"  Total merging time: {end_time - start_time:.2f} seconds")
    
    return result


def _merge_nearby_intersections(
    intersection_points, distances, tolerance=1.0, debug=False
):
    """
    A highly optimized, fully vectorized approach to merge nearby intersection points.
    Uses voxel-based clustering and vectorized operations to avoid loops.

    Args:
        intersection_points (torch.Tensor): Intersection points (N, 3)
        distances (torch.Tensor): Distances between the lines (N)
        tolerance (float): Tolerance for merging
        debug (bool): Whether to print debug information

    Returns:
        torch.Tensor: Merged intersection points
    """
    device = intersection_points.device
    n = intersection_points.size(0)

    if debug:
        stage_time = time.time()

    # 1. First, use a voxel grid to cluster points that are definitely close
    # Choose voxel size to be slightly smaller than tolerance
    voxel_size = tolerance * 0.5

    # Get min and max coordinates to define the voxel grid
    min_coords, _ = torch.min(intersection_points, dim=0)
    max_coords, _ = torch.max(intersection_points, dim=0)

    # Add padding to ensure all points are inside
    min_coords = min_coords - voxel_size
    max_coords = max_coords + voxel_size

    if debug:
        bounds_time = time.time()
        print(f"    Computed point bounds in {bounds_time - stage_time:.2f} seconds")

    # Convert points to voxel grid indices
    # Scale and quantize the points to voxel indices
    voxel_indices = torch.floor((intersection_points - min_coords) / voxel_size).long()

    # Determine maximum bits needed for each dimension
    grid_size = torch.floor((max_coords - min_coords) / voxel_size).long() + 1
    max_bits = torch.ceil(torch.log2(grid_size.float())).int().max().item()
    bits_per_dim = min(
        10, max_bits
    )  # Limit to 10 bits per dimension (30 bits total fits in int32)
    max_voxels_per_dim = 2**bits_per_dim

    # Clamp indices to valid range
    voxel_indices = torch.clamp(voxel_indices, 0, max_voxels_per_dim - 1)

    # Compute Morton codes directly on GPU using bit operations
    # Faster implementation that works directly on the GPU
    def split_bits_gpu(x):
        # This is a vectorized version that works on GPU tensors
        # Avoid unnecessary transfers between CPU and GPU
        x = x & 0x3FF  # Limit to 10 bits
        x = (x | (x << 16)) & 0x30000FF
        x = (x | (x << 8)) & 0x300F00F
        x = (x | (x << 4)) & 0x30C30C3
        x = (x | (x << 2)) & 0x9249249
        return x

    # Compute Morton code for each voxel directly on the GPU
    x_bits = split_bits_gpu(voxel_indices[:, 0])
    y_bits = split_bits_gpu(voxel_indices[:, 1])
    z_bits = split_bits_gpu(voxel_indices[:, 2])

    voxel_hash = x_bits | (y_bits << 1) | (z_bits << 2)

    if debug:
        hash_time = time.time()
        print(f"    Computed Morton codes in {hash_time - bounds_time:.2f} seconds")

    # Get unique voxels and the inverse mapping to original points
    unique_voxels, inverse_indices = torch.unique(voxel_hash, return_inverse=True)
    num_voxels = unique_voxels.size(0)

    if debug:
        unique_time = time.time()
        print(
            f"    Found {num_voxels} unique voxels in {unique_time - hash_time:.2f} seconds"
        )

    # 2. Compute weighted average for each voxel (initial clustering)
    # For each unique voxel, calculate the weighted average of all points in that voxel

    # We need to compute: sum(points[i] * weights[i]) / sum(weights[i]) for each voxel
    # This can be done using torch.scatter_add in a vectorized way

    # Create weights based on inverse distances
    weights = 1.0 / (distances + 1e-10)

    # Create tensors to accumulate weighted sums and weight sums
    weighted_sums = torch.zeros((num_voxels, 3), device=device)
    weight_sums = torch.zeros(num_voxels, device=device)

    # Use scatter_add to accumulate values for each voxel
    for dim in range(3):
        weighted_values = intersection_points[:, dim] * weights
        weighted_sums[:, dim].scatter_add_(0, inverse_indices, weighted_values)

    # Accumulate weights for normalization
    weight_sums.scatter_add_(0, inverse_indices, weights)

    # Normalize to get weighted averages
    voxel_centers = weighted_sums / weight_sums.unsqueeze(1)

    if debug:
        avg_time = time.time()
        print(
            f"    Computed initial voxel centers in {avg_time - unique_time:.2f} seconds"
        )

    # 3. Second-level clustering to merge nearby voxels
    # For voxels that are close to each other, merge them

    # Create a new set of voxel indices based on the voxel centers
    voxel_indices_2 = torch.floor((voxel_centers - min_coords) / tolerance).long()

    # Clamp indices for second-level clustering
    voxel_indices_2 = torch.clamp(voxel_indices_2, 0, max_voxels_per_dim - 1)

    # Compute Morton codes for second-level clustering directly on GPU
    x_bits_2 = split_bits_gpu(voxel_indices_2[:, 0])
    y_bits_2 = split_bits_gpu(voxel_indices_2[:, 1])
    z_bits_2 = split_bits_gpu(voxel_indices_2[:, 2])

    voxel_hash_2 = x_bits_2 | (y_bits_2 << 1) | (z_bits_2 << 2)

    # Get unique voxels in this new grid
    unique_voxels_2, inverse_indices_2 = torch.unique(voxel_hash_2, return_inverse=True)
    num_voxels_2 = unique_voxels_2.size(0)

    if debug:
        voxel2_time = time.time()
        print(
            f"    Found {num_voxels_2} second-level voxels in {voxel2_time - avg_time:.2f} seconds"
        )

    # Compute the new cluster centers
    # We need to compute weighted averages again, but now using the voxel centers as points
    # and the total weight of each voxel as the weight

    # Create tensors to accumulate weighted sums and weight sums
    weighted_sums_2 = torch.zeros((num_voxels_2, 3), device=device)
    weight_sums_2 = torch.zeros(num_voxels_2, device=device)

    # Use scatter_add to accumulate values for each final voxel
    for dim in range(3):
        weighted_values_2 = voxel_centers[:, dim] * weight_sums
        weighted_sums_2[:, dim].scatter_add_(0, inverse_indices_2, weighted_values_2)

    # Accumulate total weights
    weight_sums_2.scatter_add_(0, inverse_indices_2, weight_sums)

    # Normalize to get final cluster centers
    final_centers = weighted_sums_2 / weight_sums_2.unsqueeze(1)

    if debug:
        final_time = time.time()
        print(
            f"    Computed final {final_centers.shape[0]} cluster centers in {final_time - voxel2_time:.2f} seconds"
        )

    return final_centers

def backproject_hits_cuda(projection_data, theta, u_min, volume_shape, device="cuda", plane_id=None):
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


def project_sparse_volume(coords, values, volume_shape, theta, u_min, device='cuda', projection_size=None, differentiable=False):
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
        differentiable (bool): Whether to use differentiable projection mode
        
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
    
    if differentiable:
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
    else:
        # For non-differentiable mode, use rounding for better efficiency
        u = torch.round(u_coords).long()
        u = torch.clamp(u, min=0, max=u_size-1)  # Clamp to valid range
        
        # Create 2D indices for the sparse tensor
        x_indices = sorted_coords[:, 0].long()
        indices = torch.stack([x_indices, u], dim=0)
        
        # Create a sparse tensor and convert to dense
        sparse_projection = torch.sparse_coo_tensor(
            indices=indices, values=sorted_values, size=(x_size, u_size), device=device
        )
        
        # Convert to dense tensor
        projection = sparse_projection.to_dense()
    
    return projection

# Define aliases for backward compatibility
def project_volume_cuda_sparse(coords, values, volume_shape, theta, u_min, device='cuda', projection_size=None):
    """Legacy alias for project_sparse_volume (non-differentiable mode)"""
    return project_sparse_volume(coords, values, volume_shape, theta, u_min, device, projection_size, 
                               differentiable=False)

def project_sparse_volume_differentiable(coords, values, volume_shape, theta, u_min, device='cuda', projection_size=None,
                                        enable_diffusion=False, diffusion_sigma_t=1.0, diffusion_sigma_l=0.5, 
                                        attenuation_coeff=0.0):
    """Legacy alias for project_sparse_volume (differentiable mode)"""
    return project_sparse_volume(coords, values, volume_shape, theta, u_min, device, projection_size,
                               enable_diffusion=enable_diffusion, diffusion_sigma_t=diffusion_sigma_t,
                               diffusion_sigma_l=diffusion_sigma_l, attenuation_coeff=attenuation_coeff,
                               differentiable=True)

# Mark these functions as deprecated - they will be removed in a future version
def project_volume_cuda(volume, theta, u_min, device='cuda', projection_size=None):
    """
    DEPRECATED: Use project_sparse_volume instead.
    Convert dense volume to sparse and call project_sparse_volume.
    """
    # Convert to sparse representation
    non_zero_indices = torch.nonzero(volume, as_tuple=False)
    if non_zero_indices.shape[0] == 0:
        # Empty volume case
        return torch.zeros((volume.shape[0], 1), device=device)
        
    non_zero_values = volume[non_zero_indices[:, 0], non_zero_indices[:, 1], non_zero_indices[:, 2]]
    
    # Use sparse projection
    return project_sparse_volume(non_zero_indices, non_zero_values, volume.shape, theta, u_min, device, projection_size)

def project_volume_differentiable(volume, theta, u_min, device='cuda', projection_size=None):
    """
    DEPRECATED: Use project_sparse_volume with differentiable=True instead.
    Convert dense volume to sparse and call project_sparse_volume.
    """
    # Convert to sparse representation
    coords = torch.nonzero(volume, as_tuple=False)
    if coords.shape[0] == 0:
        # Empty volume case
        if projection_size is not None:
            return torch.zeros(projection_size, device=device)
        return torch.zeros((volume.shape[0], 1), device=device)
        
    values = volume[coords[:, 0], coords[:, 1], coords[:, 2]]
    
    # Use differentiable sparse projection
    return project_sparse_volume(coords, values, volume.shape, theta, u_min, device, projection_size, differentiable=True)
