import numpy as np
import torch
import time
from .line_representation import Line3D
from .spatial_partitioning import find_potential_intersections
from .cuda_kernels import (
    get_line_parameters_tensor,
    find_intersections_cuda,
    merge_nearby_intersections_cuda,
    backproject_hits_cuda,
    project_volume_cuda
)

class LineIntersectionSolver:
    """
    Solver for finding intersections of backprojected lines from multiple wire planes in LArTPC.
    """
    def __init__(self, volume_shape, tolerance=1.0, merge_tolerance=1.0, device='cuda', debug=False, plane_angles=None):
        """
        Initialize the solver.
        
        Args:
            volume_shape (tuple): Shape of the 3D volume (N, N, N)
            tolerance (float): Tolerance for intersection testing in mm
            merge_tolerance (float): Tolerance for merging nearby intersections in mm
            device (str): Device to use ('cuda' or 'cpu')
            debug (bool): Whether to print debug information
            plane_angles (dict, optional): Dictionary mapping plane_id to angle in radians.
                                          If None, uses default angles (0°, 90°, 60°, 120°).
        """
        self.volume_shape = volume_shape
        self.tolerance = tolerance
        self.merge_tolerance = merge_tolerance
        self.device = device
        self.debug = debug
        
        # Set volume bounds
        self.volume_min = np.zeros(3, dtype=np.float32)
        self.volume_max = np.array(volume_shape, dtype=np.float32)
        
        # Initialize standard wire plane angles (in radians)
        if plane_angles is None:
            # Default angles
            self.plane_angles = {
                0: 0.0,                  # 0 degrees (vertical)
                1: np.pi/2,              # 90 degrees (horizontal)
                2: np.pi/3,              # 60 degrees
                3: 2*np.pi/3             # 120 degrees
            }
        else:
            # Use custom angles
            self.plane_angles = plane_angles
            
        # Initialize u_min values for each plane
        # These ensure nonnegative indices in the projections
        self.u_min_values = {}
        self._compute_u_min_values()
        
        # Calculate maximum u-coordinate for each plane to standardize projection sizes
        self._compute_projection_dimensions()
    
    def _compute_u_min_values(self):
        """
        Compute u_min values for each plane to ensure nonnegative indices in projections.
        This should be called whenever plane angles are updated.
        """
        self.u_min_values = {}
        for plane_id, theta in self.plane_angles.items():
            # Compute maximum possible u-coordinate for this plane
            sin_theta = np.sin(theta)
            cos_theta = np.cos(theta)
            
            # Compute corners of the volume
            corners = []
            for x in [0, self.volume_shape[0]-1]:
                for y in [0, self.volume_shape[1]-1]:
                    for z in [0, self.volume_shape[2]-1]:
                        corners.append(np.array([x, y, z]))
            
            # Compute u for each corner
            u_values = [-sin_theta * corner[1] + cos_theta * corner[2] for corner in corners]
            
            # Set u_min to ensure nonnegative indices
            self.u_min_values[plane_id] = np.floor(min(u_values))
    
    def _compute_projection_dimensions(self):
        """
        Compute standardized projection dimensions for each plane based on volume shape and wire orientation.
        This ensures that all projections have consistent sizes for direct comparison.
        """
        self.projection_sizes = {}
        
        # Get volume dimensions
        N = self.volume_shape[0]
        
        for plane_id, theta in self.plane_angles.items():
            # Calculate u-coordinates for the volume corners
            sin_theta = np.sin(theta)
            cos_theta = np.cos(theta)
            
            # Calculate u-coordinate for the four corners of the (y,z) plane
            corners = [
                (-sin_theta * 0 + cos_theta * 0),             # (0,0)
                (-sin_theta * (N-1) + cos_theta * 0),         # (N-1,0)
                (-sin_theta * 0 + cos_theta * (N-1)),         # (0,N-1)
                (-sin_theta * (N-1) + cos_theta * (N-1))      # (N-1,N-1)
            ]
            
            # Find the min and max u-coordinates
            min_u = min(corners)
            max_u = max(corners)
            
            # Adjust by u_min to get the actual projection range
            u_min = self.u_min_values[plane_id]
            min_u_adj = min_u - u_min
            max_u_adj = max_u - u_min
            
            # Calculate the standard projection size (x dimension is always the volume size)
            # Add buffer to ensure we have enough space for all points
            projection_width = int(np.ceil(max_u_adj)) + 2  # Add buffer of 2
            
            self.projection_sizes[plane_id] = (N, projection_width)
            
            if self.debug:
                print(f"Plane {plane_id}: theta={theta:.2f}, projection size={self.projection_sizes[plane_id]}")
                print(f"  u range: [{min_u_adj:.2f}, {max_u_adj:.2f}]")
    
    def set_plane_angles(self, plane_angles):
        """
        Set wire plane angles.
        
        Args:
            plane_angles (dict): Dictionary mapping plane_id to wire plane angle in radians
        """
        self.plane_angles = plane_angles
        
        # Recompute u_min values and projection dimensions when angles change
        self._compute_u_min_values()
        self._compute_projection_dimensions()
    
    def project_volume_cuda(self, volume, theta, u_min, device=None, projection_size=None):
        """
        Project a 3D volume to a 2D projection using CUDA.
        Automatically converts dense volumes to sparse format for efficiency.
        
        Args:
            volume (torch.Tensor): 3D volume of shape (N, N, N)
            theta (float): Angle of the wire plane in radians
            u_min (float): Minimum u-coordinate
            device (str): Device to use ('cuda' or 'cpu')
            projection_size (tuple): Optional. Size of the output projection (depth, width).
                                   If None, it will be determined from plane angles.
            
        Returns:
            torch.Tensor: 2D projection of shape (N, U)
        """
        if device is None:
            device = self.device
        
        # Find the plane ID for this theta value to get the standardized projection size
        plane_id = None
        for pid, angle in self.plane_angles.items():
            if abs(angle - theta) < 1e-6:  # Close enough to be considered the same angle
                plane_id = pid
                break
        
        # Get the standardized projection size if we found a matching plane ID and it's not already provided
        if projection_size is None:
            if plane_id is not None and hasattr(self, 'projection_sizes'):
                projection_size = self.projection_sizes[plane_id]
        
        # Convert to sparse representation for efficiency
        coords = torch.nonzero(volume, as_tuple=False)
        if coords.shape[0] > 0:
            values = volume[coords[:, 0], coords[:, 1], coords[:, 2]]
        else:
            # Empty volume case
            values = torch.tensor([], device=device)
            
        # Import the unified sparse projection function
        from .cuda_kernels import project_sparse_volume
        
        # Project the volume using sparse representation
        return project_sparse_volume(
            coords=coords,
            values=values,
            volume_shape=volume.shape,
            theta=theta,
            u_min=u_min,
            device=device,
            projection_size=projection_size,
            differentiable=False  # Use non-differentiable mode for standard projection
        )
    
    def backproject_plane(self, projection_data, plane_id):
        """
        Backproject hits from a wire plane into 3D lines.
        
        Args:
            projection_data (torch.Tensor): 2D projection data (x, u)
            plane_id (int): ID of the wire plane
            
        Returns:
            tuple: (points, directions, plane_ids) representing the backprojected lines
        """
        theta = self.plane_angles[plane_id]
        u_min = self.u_min_values[plane_id]
        
        if self.debug:
            start_time = time.time()
            print(f"Backprojecting plane {plane_id} (theta={theta:.2f})...")
            print(f"  Projection shape: {projection_data.shape}")
            print(f"  Non-zero hits: {torch.count_nonzero(projection_data).item()}")
        
        points, directions, plane_ids = backproject_hits_cuda(
            projection_data,
            theta,
            u_min,
            self.volume_shape,
            device=self.device,
            plane_id=plane_id
        )
        
        if self.debug:
            end_time = time.time()
            print(f"  Backprojection complete in {end_time - start_time:.2f} seconds")
            print(f"  Generated {points.shape[0]} lines")
        
        return points, directions, plane_ids
    
    def find_intersections(self, projections, fast_merge=True, snap_to_grid=True):
        """
        Find intersections between backprojected lines from multiple wire planes.
        
        Args:
            projections (dict): Dictionary mapping plane_id to projection data
            fast_merge (bool): Whether to use fast merge mode for clustering (much faster but slightly less precise)
            snap_to_grid (bool): Whether to snap intersection points to the nearest grid points
            
        Returns:
            torch.Tensor: Intersection points (N, 3)
        """
        if self.debug:
            total_start_time = time.time()
            print(f"Finding intersections between lines from {len(projections)} planes...")
        
        # Convert projections to lines
        all_points = []
        all_directions = []
        all_plane_ids = []
        
        if self.debug:
            print("Converting projections to lines...")
            backproject_start = time.time()
        
        for plane_id, projection in projections.items():
            points, directions, plane_ids = self.backproject_plane(projection, plane_id)
            all_points.append(points)
            all_directions.append(directions)
            all_plane_ids.append(plane_ids)
        
        # Concatenate all lines
        all_points = torch.cat(all_points, dim=0)
        all_directions = torch.cat(all_directions, dim=0)
        all_plane_ids = torch.cat(all_plane_ids, dim=0)
        
        if self.debug:
            backproject_end = time.time()
            print(f"  Converted all projections to {all_points.shape[0]} lines in {backproject_end - backproject_start:.2f} seconds")
        
        # Find all pairwise intersections
        intersection_points = []
        intersection_distances = []
        
        if self.debug:
            print("Finding pairwise intersections between planes...")
            pairwise_start = time.time()
        
        # Find intersections between lines from different planes
        num_planes = len(projections)
        plane_ids = list(projections.keys())
        
        for i in range(num_planes):
            for j in range(i + 1, num_planes):
                plane_id1 = plane_ids[i]
                plane_id2 = plane_ids[j]
                
                if self.debug:
                    plane_pair_start = time.time()
                    print(f"  Processing plane pair ({plane_id1}, {plane_id2})...")
                
                # Get line parameters for each plane
                mask1 = all_plane_ids == plane_id1
                mask2 = all_plane_ids == plane_id2
                
                points1 = all_points[mask1]
                directions1 = all_directions[mask1]
                plane_ids1 = all_plane_ids[mask1]
                
                points2 = all_points[mask2]
                directions2 = all_directions[mask2]
                plane_ids2 = all_plane_ids[mask2]
                
                if self.debug:
                    print(f"    Plane {plane_id1}: {points1.shape[0]} lines")
                    print(f"    Plane {plane_id2}: {points2.shape[0]} lines")
                    print(f"    Potential pairs: {points1.shape[0] * points2.shape[0]}")
                    intersection_find_start = time.time()
                
                # Find intersections
                points, indices1, indices2, distances = find_intersections_cuda(
                    points1, directions1, plane_ids1,
                    points2, directions2, plane_ids2,
                    self.tolerance, self.device, debug=self.debug,
                    snap_to_grid=snap_to_grid
                )
                
                if self.debug:
                    intersection_find_end = time.time()
                    print(f"    Found {points.shape[0]} intersections in {intersection_find_end - intersection_find_start:.2f} seconds")
                
                intersection_points.append(points)
                intersection_distances.append(distances)
                
                if self.debug:
                    plane_pair_end = time.time()
                    print(f"  Finished plane pair in {plane_pair_end - plane_pair_start:.2f} seconds")
        
        if self.debug:
            pairwise_end = time.time()
            print(f"  Completed all pairwise intersections in {pairwise_end - pairwise_start:.2f} seconds")
        
        # Concatenate all intersection points
        if intersection_points:
            if self.debug:
                merge_start = time.time()
                print(f"Merging intersection points...")
                num_intersections = sum(p.shape[0] for p in intersection_points)
                print(f"  Total intersections before merging: {num_intersections}")
            
            all_intersection_points = torch.cat(intersection_points, dim=0)
            all_distances = torch.cat(intersection_distances, dim=0)
            
            # Merge nearby intersections
            merged_points = merge_nearby_intersections_cuda(
                all_intersection_points,
                all_distances,
                self.merge_tolerance,
                debug=self.debug,
                fast_mode=fast_merge
            )
            
            if self.debug:
                merge_end = time.time()
                print(f"  Merged {all_intersection_points.shape[0]} points into {merged_points.shape[0]} clusters")
                print(f"  Merging completed in {merge_end - merge_start:.2f} seconds")
                
                total_end = time.time()
                print(f"Total intersection finding time: {total_end - total_start_time:.2f} seconds")
            
            return merged_points
        else:
            if self.debug:
                print("No intersections found between any planes")
                total_end = time.time()
                print(f"Total intersection finding time: {total_end - total_start_time:.2f} seconds")
            
            return torch.zeros((0, 3), device=self.device)

    def solve_inverse_problem(self, projections, fast_merge=True, snap_to_grid=True):
        """
        Solve the inverse problem: find 3D points that are consistent with
        intersections of backprojected lines from different wire planes.
        
        Args:
            projections (dict): Dictionary mapping plane_id to projection data
            fast_merge (bool): Whether to use fast merge mode for clustering (much faster but slightly less precise)
            snap_to_grid (bool): Whether to snap intersection points to the nearest grid points
            
        Returns:
            torch.Tensor: 3D points that are consistent with the projections
        """
        if self.debug:
            start_time = time.time()
            print("Solving inverse problem...")
            if fast_merge:
                print("  Using FAST merge mode for maximum speed")
            else:
                print("  Using PRECISE merge mode (slower but more accurate)")
            if snap_to_grid:
                print("  Snapping intersection points to nearest grid points")
        
        # Find intersections between backprojected lines
        intersection_points = self.find_intersections(projections, fast_merge, snap_to_grid)
        
        # Filter out points outside the volume bounds
        if intersection_points.size(0) > 0:
            # Create boundary conditions for the volume
            volume_min = torch.zeros(3, device=self.device)
            volume_max = torch.tensor(self.volume_shape, device=self.device) - 1
            
            # Check which points are within bounds
            in_bounds = torch.all(
                (intersection_points >= volume_min) & 
                (intersection_points <= volume_max),
                dim=1
            )
            
            # Filter points
            points_in_bounds = intersection_points[in_bounds]
            
            if self.debug:
                filtered_count = intersection_points.size(0) - points_in_bounds.size(0)
                if filtered_count > 0:
                    print(f"  Filtered out {filtered_count} points outside volume bounds")
                print(f"  Final reconstructed points: {points_in_bounds.size(0)}")
            
            return points_in_bounds
        else:
            if self.debug:
                print("  No intersections found")
            return torch.zeros((0, 3), device=self.device)
    
    def lines_from_numpy_array(self, array, plane_id):
        """
        Convert a numpy array of line parameters to Line3D objects.
        
        Args:
            array (np.ndarray): Array of shape (N, 6) containing line parameters
                               (point_x, point_y, point_z, dir_x, dir_y, dir_z)
            plane_id (int): ID of the wire plane
            
        Returns:
            list: List of Line3D objects
        """
        lines = []
        for i in range(array.shape[0]):
            point = array[i, 0:3]
            direction = array[i, 3:6]
            line = Line3D(point, direction, plane_id)
            lines.append(line)
        
        return lines
    
    def intersect_lines_cpu(self, lines_by_plane, snap_to_grid=True):
        """
        Find intersections between lines from different planes using CPU.
        This is a slower alternative to the CUDA implementation.
        
        Args:
            lines_by_plane (dict): Dictionary mapping plane_id to list of Line3D objects
            snap_to_grid (bool): Whether to snap intersection points to the nearest grid points
            
        Returns:
            np.ndarray: Intersection points (N, 3)
        """
        # Use spatial partitioning to find potential intersections
        candidates = find_potential_intersections(
            lines_by_plane,
            self.tolerance,
            (self.volume_min, self.volume_max)
        )
        
        # Compute actual intersections
        intersection_points = []
        distances = []
        
        for line1, line2 in candidates:
            # Skip lines from the same plane
            if line1.plane_id == line2.plane_id:
                continue
            
            # Compute closest points between the lines
            # This is done by solving a 2x2 linear system
            
            # Get line parameters
            p1 = line1.point
            d1 = line1.direction
            p2 = line2.point
            d2 = line2.direction
            
            # Compute coefficients of the 2x2 linear system
            # [a, b; c, d] * [t1; t2] = [e; f]
            
            a = np.dot(d1, d1)
            b = -np.dot(d1, d2)
            c = b
            d = np.dot(d2, d2)
            
            dp = p2 - p1
            e = np.dot(dp, d1)
            f = -np.dot(dp, d2)
            
            # Compute determinant
            det = a * d - b * c
            
            # Handle parallel lines
            if abs(det) < 1e-10:
                continue
            
            # Solve for t1 and t2
            t1 = (d * e - b * f) / det
            t2 = (a * f - c * e) / det
            
            # Compute closest points
            closest1 = p1 + t1 * d1
            closest2 = p2 + t2 * d2
            
            # Compute distance between closest points
            distance = np.linalg.norm(closest1 - closest2)
            
            # If distance is below tolerance, consider it an intersection
            if distance <= self.tolerance:
                # Compute intersection point as midpoint of closest points
                intersection = (closest1 + closest2) / 2
                
                # Snap to grid if requested
                if snap_to_grid:
                    intersection = np.round(intersection)
                
                intersection_points.append(intersection)
                distances.append(distance)
        
        # Convert to numpy arrays
        if intersection_points:
            intersection_points = np.array(intersection_points)
            distances = np.array(distances)
            
            # Merge nearby intersections (simple clustering)
            # This is a simplified version of the CUDA implementation
            from sklearn.cluster import DBSCAN
            
            if intersection_points.shape[0] > 1:
                # Use DBSCAN for clustering
                clustering = DBSCAN(eps=self.merge_tolerance, min_samples=1).fit(intersection_points)
                labels = clustering.labels_
                
                # Compute cluster centers
                unique_labels = np.unique(labels)
                merged_points = np.zeros((len(unique_labels), 3), dtype=np.float32)
                
                for i, label in enumerate(unique_labels):
                    mask = labels == label
                    points = intersection_points[mask]
                    point_distances = distances[mask]
                    
                    # Use inverse distances as weights
                    weights = 1.0 / (point_distances + 1e-10)
                    weights /= np.sum(weights)
                    
                    # Compute weighted average
                    merged_points[i] = np.sum(points * weights[:, np.newaxis], axis=0)
                
                return merged_points
            else:
                return intersection_points
        else:
            return np.zeros((0, 3), dtype=np.float32) 