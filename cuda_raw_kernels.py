import numpy as np
import numba
from numba import cuda
import math

@cuda.jit
def closest_points_between_lines_kernel(points1, directions1, points2, directions2, 
                                       plane_ids1, plane_ids2, tolerance,
                                       out_intersection_points, out_distances):
    """
    CUDA kernel to compute the closest points between two sets of lines.
    
    Args:
        points1 (device array): Points on the first set of lines (N, 3)
        directions1 (device array): Direction vectors of the first set of lines (N, 3)
        points2 (device array): Points on the second set of lines (M, 3)
        directions2 (device array): Direction vectors of the second set of lines (M, 3)
        plane_ids1 (device array): Plane IDs for the first set of lines (N)
        plane_ids2 (device array): Plane IDs for the second set of lines (M)
        tolerance (float): Tolerance for intersection testing
        out_intersection_points (device array): Output array for intersection points (N * M, 3)
        out_distances (device array): Output array for distances (N * M)
    """
    # Get thread indices
    i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    j = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
    
    # Check bounds
    if i < points1.shape[0] and j < points2.shape[0]:
        # Skip lines from the same plane
        if plane_ids1[i] == plane_ids2[j]:
            # Mark as invalid by setting distance to a large value
            out_distances[i * points2.shape[0] + j] = 1e10
            return
        
        # Get line parameters
        p1_x = points1[i, 0]
        p1_y = points1[i, 1]
        p1_z = points1[i, 2]
        
        d1_x = directions1[i, 0]
        d1_y = directions1[i, 1]
        d1_z = directions1[i, 2]
        
        p2_x = points2[j, 0]
        p2_y = points2[j, 1]
        p2_z = points2[j, 2]
        
        d2_x = directions2[j, 0]
        d2_y = directions2[j, 1]
        d2_z = directions2[j, 2]
        
        # Compute coefficients of the 2x2 linear system
        # [a, b; c, d] * [t1; t2] = [e; f]
        
        a = d1_x * d1_x + d1_y * d1_y + d1_z * d1_z
        b = -(d1_x * d2_x + d1_y * d2_y + d1_z * d2_z)
        c = b
        d = d2_x * d2_x + d2_y * d2_y + d2_z * d2_z
        
        dp_x = p2_x - p1_x
        dp_y = p2_y - p1_y
        dp_z = p2_z - p1_z
        
        e = dp_x * d1_x + dp_y * d1_y + dp_z * d1_z
        f = -(dp_x * d2_x + dp_y * d2_y + dp_z * d2_z)
        
        # Compute determinant
        det = a * d - b * c
        
        # Initialize t values
        t1 = 0.0
        t2 = 0.0
        
        # Solve for t1 and t2 if not parallel
        if abs(det) > 1e-10:
            t1 = (d * e - b * f) / det
            t2 = (a * f - c * e) / det
        
        # Compute closest points
        closest1_x = p1_x + t1 * d1_x
        closest1_y = p1_y + t1 * d1_y
        closest1_z = p1_z + t1 * d1_z
        
        closest2_x = p2_x + t2 * d2_x
        closest2_y = p2_y + t2 * d2_y
        closest2_z = p2_z + t2 * d2_z
        
        # Compute distance between closest points
        dx = closest1_x - closest2_x
        dy = closest1_y - closest2_y
        dz = closest1_z - closest2_z
        
        distance = math.sqrt(dx * dx + dy * dy + dz * dz)
        
        # If distance is below tolerance, consider it an intersection
        if distance <= tolerance:
            # Compute intersection point as midpoint
            idx = i * points2.shape[0] + j
            out_intersection_points[idx, 0] = (closest1_x + closest2_x) / 2.0
            out_intersection_points[idx, 1] = (closest1_y + closest2_y) / 2.0
            out_intersection_points[idx, 2] = (closest1_z + closest2_z) / 2.0
            out_distances[idx] = distance
        else:
            # Mark as invalid by setting distance to a large value
            out_distances[i * points2.shape[0] + j] = 1e10

@cuda.jit
def backproject_hits_kernel(projection_data, hits_x, hits_u, 
                          sin_theta, cos_theta, u_min,
                          points, directions, plane_ids, plane_id):
    """
    CUDA kernel to backproject 2D hits into 3D lines.
    
    Args:
        projection_data (device array): 2D projection data (x, u)
        hits_x (device array): x-coordinates of hits
        hits_u (device array): u-coordinates of hits
        sin_theta (float): Sine of the angle of the wire plane
        cos_theta (float): Cosine of the angle of the wire plane
        u_min (float): Minimum u-coordinate
        points (device array): Output array for points on the lines (N, 3)
        directions (device array): Output array for direction vectors (N, 3)
        plane_ids (device array): Output array for plane IDs (N)
        plane_id (int): ID of the wire plane
    """
    i = cuda.grid(1)
    
    if i < hits_x.shape[0]:
        x = hits_x[i]
        u = hits_u[i]
        
        # Compute u in original space
        u_orig = float(u) + u_min
        
        # Initialize point
        points[i, 0] = float(x)
        
        # For the vertical case (theta = 0), u = z
        if abs(sin_theta) < 1e-10:
            points[i, 1] = 0.0  # y = 0
            points[i, 2] = u_orig  # z = u
        # For the horizontal case (theta = 90 degrees), u = -y
        elif abs(cos_theta) < 1e-10:
            points[i, 1] = -u_orig  # y = -u
            points[i, 2] = 0.0  # z = 0
        # For the general case
        else:
            # Choose a point where either y=0 or z=0
            points[i, 1] = 0.0  # y = 0
            points[i, 2] = u_orig / cos_theta  # z = u / cos(theta)
        
        # Compute direction vectors (perpendicular to the wire direction)
        directions[i, 0] = 0.0  # Perpendicular to x (the drift direction)
        directions[i, 1] = cos_theta  # y-component
        directions[i, 2] = sin_theta  # z-component
        
        # Normalize direction
        norm = math.sqrt(directions[i, 1] * directions[i, 1] + directions[i, 2] * directions[i, 2])
        if norm > 0:
            directions[i, 1] /= norm
            directions[i, 2] /= norm
        
        # Set plane ID
        plane_ids[i] = plane_id

@cuda.jit
def find_hits_kernel(projection_data, threshold, out_hits_count, out_hits_x, out_hits_u):
    """
    CUDA kernel to find non-zero hits in a projection.
    
    Args:
        projection_data (device array): 2D projection data (x, u)
        threshold (float): Threshold for considering a hit
        out_hits_count (device array): Output array for the number of hits found
        out_hits_x (device array): Output array for x-coordinates of hits
        out_hits_u (device array): Output array for u-coordinates of hits
    """
    i = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    j = cuda.threadIdx.y + cuda.blockIdx.y * cuda.blockDim.y
    
    if i < projection_data.shape[0] and j < projection_data.shape[1]:
        if projection_data[i, j] > threshold:
            # Atomic increment to keep track of the number of hits
            idx = cuda.atomic.add(out_hits_count, 0, 1)
            
            # Store hit coordinates if there's space
            if idx < out_hits_x.shape[0]:
                out_hits_x[idx] = i
                out_hits_u[idx] = j

def find_hit_indices(projection_data, threshold=0.0):
    """
    Find non-zero hits in a projection.
    
    Args:
        projection_data (numpy.ndarray): 2D projection data (x, u)
        threshold (float): Threshold for considering a hit
        
    Returns:
        tuple: (hits_x, hits_u) - Arrays of x and u coordinates of hits
    """
    # Use NumPy to find non-zero elements
    hits = np.array(np.where(projection_data > threshold)).T
    hits_x = hits[:, 0].astype(np.int32)
    hits_u = hits[:, 1].astype(np.int32)
    
    return hits_x, hits_u

def backproject_hits_cuda_raw(projection_data, theta, u_min, plane_id):
    """
    Backproject 2D hits into 3D lines using CUDA.
    
    Args:
        projection_data (numpy.ndarray): 2D projection data (x, u)
        theta (float): Angle of the wire plane in radians
        u_min (float): Minimum u-coordinate
        plane_id (int): ID of the wire plane
        
    Returns:
        tuple: (points, directions, plane_ids) - Arrays representing the backprojected lines
    """
    # Find hits in the projection
    hits_x, hits_u = find_hit_indices(projection_data)
    
    # If no hits, return empty arrays
    if len(hits_x) == 0:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.float32), np.zeros(0, dtype=np.int32)
    
    # Calculate sin and cos of the angle
    sin_theta = math.sin(theta)
    cos_theta = math.cos(theta)
    
    # Allocate output arrays
    points = np.zeros((len(hits_x), 3), dtype=np.float32)
    directions = np.zeros((len(hits_x), 3), dtype=np.float32)
    plane_ids_array = np.zeros(len(hits_x), dtype=np.int32)
    
    # Copy data to device
    d_hits_x = cuda.to_device(hits_x)
    d_hits_u = cuda.to_device(hits_u)
    d_points = cuda.to_device(points)
    d_directions = cuda.to_device(directions)
    d_plane_ids = cuda.to_device(plane_ids_array)
    
    # Launch kernel
    threads_per_block = 256
    blocks = (len(hits_x) + threads_per_block - 1) // threads_per_block
    
    backproject_hits_kernel[blocks, threads_per_block](
        projection_data, d_hits_x, d_hits_u, 
        sin_theta, cos_theta, u_min,
        d_points, d_directions, d_plane_ids, plane_id
    )
    
    # Copy results back to host
    points = d_points.copy_to_host()
    directions = d_directions.copy_to_host()
    plane_ids_array = d_plane_ids.copy_to_host()
    
    return points, directions, plane_ids_array

def find_line_intersections_cuda_raw(points1, directions1, plane_ids1,
                                     points2, directions2, plane_ids2,
                                     tolerance=1.0):
    """
    Find intersections between two sets of lines using CUDA.
    
    Args:
        points1 (numpy.ndarray): Points on the first set of lines (N, 3)
        directions1 (numpy.ndarray): Direction vectors of the first set of lines (N, 3)
        plane_ids1 (numpy.ndarray): Plane IDs for the first set of lines (N)
        points2 (numpy.ndarray): Points on the second set of lines (M, 3)
        directions2 (numpy.ndarray): Direction vectors of the second set of lines (M, 3)
        plane_ids2 (numpy.ndarray): Plane IDs for the second set of lines (M)
        tolerance (float): Tolerance for intersection testing
        
    Returns:
        tuple: (intersection_points, distances) - Arrays of intersection points and distances
    """
    # Get sizes
    n = points1.shape[0]
    m = points2.shape[0]
    
    # If either set is empty, return empty arrays
    if n == 0 or m == 0:
        return np.zeros((0, 3), dtype=np.float32), np.zeros(0, dtype=np.float32)
    
    # Allocate output arrays
    intersection_points = np.zeros((n * m, 3), dtype=np.float32)
    distances = np.ones(n * m, dtype=np.float32) * 1e10  # Initialize to large value
    
    # Copy data to device
    d_points1 = cuda.to_device(points1)
    d_directions1 = cuda.to_device(directions1)
    d_plane_ids1 = cuda.to_device(plane_ids1)
    d_points2 = cuda.to_device(points2)
    d_directions2 = cuda.to_device(directions2)
    d_plane_ids2 = cuda.to_device(plane_ids2)
    d_intersection_points = cuda.to_device(intersection_points)
    d_distances = cuda.to_device(distances)
    
    # Launch kernel with 2D grid
    threads_per_block = (16, 16)
    blocks = ((n + threads_per_block[0] - 1) // threads_per_block[0],
              (m + threads_per_block[1] - 1) // threads_per_block[1])
    
    closest_points_between_lines_kernel[blocks, threads_per_block](
        d_points1, d_directions1, d_points2, d_directions2,
        d_plane_ids1, d_plane_ids2, tolerance,
        d_intersection_points, d_distances
    )
    
    # Copy results back to host
    distances = d_distances.copy_to_host()
    intersection_points = d_intersection_points.copy_to_host()
    
    # Filter valid intersections
    valid_indices = np.where(distances < tolerance)[0]
    valid_points = intersection_points[valid_indices]
    valid_distances = distances[valid_indices]
    
    return valid_points, valid_distances 