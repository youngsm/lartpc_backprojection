import numpy as np
import torch
import math

def get_line_parameters_tensor(lines, device='cuda'):
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
    plane_ids = np.array([line.plane_id if line.plane_id is not None else -1 for line in lines], dtype=np.int32)
    
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
    t1[non_parallel_mask] = (d[non_parallel_mask] * e[non_parallel_mask] - b[non_parallel_mask] * f[non_parallel_mask]) / det[non_parallel_mask]
    t2[non_parallel_mask] = (a[non_parallel_mask] * f[non_parallel_mask] - c[non_parallel_mask] * e[non_parallel_mask]) / det[non_parallel_mask]
    
    # Compute closest points
    # p1 + t1 * d1 and p2 + t2 * d2
    t1 = t1.unsqueeze(2).expand(n, m, 3)  # (n, m, 3)
    t2 = t2.unsqueeze(2).expand(n, m, 3)  # (n, m, 3)
    
    closest1 = p1 + t1 * d1  # (n, m, 3)
    closest2 = p2 + t2 * d2  # (n, m, 3)
    
    # Compute distances between closest points
    distances = torch.sqrt(torch.sum((closest1 - closest2) ** 2, dim=2))  # (n, m)
    
    return closest1, closest2, distances

def find_intersections_cuda(points1, directions1, plane_ids1, 
                           points2, directions2, plane_ids2, 
                           tolerance=1.0, device='cuda'):
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
        
    Returns:
        tuple: (intersection_points, line_indices1, line_indices2, distances)
    """
    # Compute closest points between all pairs of lines
    closest1, closest2, distances = closest_points_between_lines_cuda(points1, directions1, points2, directions2)
    
    # Create masks for valid intersections
    # 1. Distance is below tolerance
    distance_mask = distances <= tolerance
    
    # 2. Lines are from different planes
    plane_mask = plane_ids1.unsqueeze(1) != plane_ids2.unsqueeze(0)
    
    # 3. Combined mask
    valid_mask = distance_mask & plane_mask
    
    # Count valid intersections
    num_intersections = torch.sum(valid_mask).item()
    
    # Get indices of valid intersections
    indices = torch.nonzero(valid_mask, as_tuple=True)
    line_indices1 = indices[0]
    line_indices2 = indices[1]
    
    # Compute intersection points as midpoints of closest points
    intersection_points = (closest1[valid_mask] + closest2[valid_mask]) / 2.0
    
    # Get distances of valid intersections
    valid_distances = distances[valid_mask]
    
    return intersection_points, line_indices1, line_indices2, valid_distances

def merge_nearby_intersections_cuda(intersection_points, distances, tolerance=1.0):
    """
    Merge intersection points that are very close to each other.
    
    Args:
        intersection_points (torch.Tensor): Intersection points (N, 3)
        distances (torch.Tensor): Distances between the lines (N)
        tolerance (float): Tolerance for merging
        
    Returns:
        torch.Tensor: Merged intersection points
    """
    # If no intersections, return empty tensor
    if intersection_points.size(0) == 0:
        return torch.zeros((0, 3), device=intersection_points.device)
    
    # Initialize cluster assignments
    n = intersection_points.size(0)
    cluster_assignments = torch.arange(n, device=intersection_points.device)
    
    # Compute pairwise distances between all intersection points
    pairwise_distances = torch.cdist(intersection_points, intersection_points)
    
    # Find pairs that are closer than tolerance
    close_pairs = torch.nonzero(pairwise_distances <= tolerance, as_tuple=True)
    
    # For each close pair, assign both points to the smaller cluster index
    for i, j in zip(*close_pairs):
        if i != j:
            old_cluster = torch.max(cluster_assignments[i], cluster_assignments[j])
            new_cluster = torch.min(cluster_assignments[i], cluster_assignments[j])
            cluster_assignments = torch.where(cluster_assignments == old_cluster, new_cluster, cluster_assignments)
    
    # Count unique clusters
    unique_clusters = torch.unique(cluster_assignments)
    num_clusters = unique_clusters.size(0)
    
    # Initialize merged points tensor
    merged_points = torch.zeros((num_clusters, 3), device=intersection_points.device)
    
    # For each cluster, compute the weighted average of its points
    # Points with smaller distances (better intersections) have higher weights
    for i, cluster in enumerate(unique_clusters):
        mask = cluster_assignments == cluster
        cluster_points = intersection_points[mask]
        cluster_distances = distances[mask]
        
        # Use inverse distances as weights
        weights = 1.0 / (cluster_distances + 1e-10)
        weights = weights / torch.sum(weights)
        
        # Compute weighted average
        weighted_points = cluster_points * weights.unsqueeze(1)
        merged_points[i] = torch.sum(weighted_points, dim=0)
    
    return merged_points

def backproject_hits_cuda(projection_data, theta, u_min, volume_shape, device='cuda'):
    """
    Backproject 2D hits into 3D lines using CUDA.
    
    Args:
        projection_data (torch.Tensor): 2D projection data (x, u)
        theta (float): Angle of the wire plane in radians
        u_min (float): Minimum u-coordinate
        volume_shape (tuple): Shape of the 3D volume (N, N, N)
        device (str): Device to use ('cuda' or 'cpu')
        
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
    
    # Set plane ID
    plane_id = torch.ones(n_hits, device=device, dtype=torch.int32) * (0 if theta == 0 else (1 if theta == math.pi/2 else 2))
    
    return points, directions, plane_id

def project_volume_cuda(volume, theta, u_min, device='cuda'):
    """
    Project a 3D volume to a 2D projection using CUDA.
    
    Args:
        volume (torch.Tensor): 3D volume of shape (N, N, N)
        theta (float): Angle of the wire plane in radians
        u_min (float): Minimum u-coordinate
        device (str): Device to use ('cuda' or 'cpu')
        
    Returns:
        torch.Tensor: 2D projection of shape (N, U) where U depends on the projection
    """
    # Get volume shape
    N = volume.shape[0]
    
    # Create meshgrid for the volume
    x, y, z = torch.meshgrid(
        torch.arange(N, device=device),
        torch.arange(N, device=device),
        torch.arange(N, device=device),
        indexing='ij'
    )
    
    # Flatten coordinates
    x_flat = x.flatten()
    y_flat = y.flatten()
    z_flat = z.flatten()
    values_flat = volume.flatten()
    
    # Compute u-coordinates
    sin_theta = math.sin(theta)
    cos_theta = math.cos(theta)
    u_flat = torch.round(-sin_theta * y_flat + cos_theta * z_flat) - u_min
    u_flat = u_flat.long()
    
    # Determine the size of the projection
    u_max = int(torch.max(u_flat).item()) + 1
    
    # Create the projection tensor
    projection = torch.zeros((N, u_max), device=device)
    
    # Use scatter_add to sum the values along the projection lines
    projection_idx = torch.stack([x_flat, u_flat], dim=1)
    projection.scatter_add_(0, projection_idx, values_flat)
    
    return projection 