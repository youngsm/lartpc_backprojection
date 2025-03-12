import numpy as np
import torch
import time
from .cuda_kernels import (
    find_intersections_between_lines_cuda,
    backproject_hits_into_lines_cuda,
)

class LineIntersectionSolver:
    """
    Solver for finding intersections of backprojected lines from multiple wire planes in LArTPC.
    """
    def __init__(self, volume_shape, intersection_tolerance=1.0, device='cuda', debug=False, plane_angles=None):
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
        self.intersection_tolerance = intersection_tolerance
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
        
    def backproject_plane_into_lines(self, projection_data, plane_id):
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
        
        points, directions, plane_ids = backproject_hits_into_lines_cuda(
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
    
    def backproject_planes_to_3d(self, projections, snap_to_grid=True, voxel_size=1.0):
        """
        Find intersections between backprojected lines from multiple wire planes.
        
        Args:
            projections (dict): Dictionary mapping plane_id to projection data
            snap_to_grid (bool): Whether to snap intersection points to the nearest grid points
            voxel_size (float): Size of the voxels in the volume
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
            points, directions, plane_ids = self.backproject_plane_into_lines(projection, plane_id)
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
                points, indices1, indices2 = (
                    find_intersections_between_lines_cuda(
                        points1,directions1,plane_ids1,
                        points2,directions2,plane_ids2,
                        self.intersection_tolerance,
                        self.device,
                        debug=self.debug,
                        snap_to_grid=snap_to_grid,
                        voxel_size=voxel_size,
                    )
                )
                
                if self.debug:
                    intersection_find_end = time.time()
                    print(f"    Found {points.shape[0]} intersections in {intersection_find_end - intersection_find_start:.2f} seconds")
                
                intersection_points.append(points)
                
                if self.debug:
                    plane_pair_end = time.time()
                    print(f"  Finished plane pair in {plane_pair_end - plane_pair_start:.2f} seconds")
        
        if self.debug:
            pairwise_end = time.time()
            print(f"  Completed all pairwise intersections in {pairwise_end - pairwise_start:.2f} seconds")
        
        if intersection_points:
            return torch.cat(intersection_points, dim=0)
        else:
            if self.debug:
                print("No intersections found between any planes")
                total_end = time.time()
                print(f"Total intersection finding time: {total_end - total_start_time:.2f} seconds")
            
            return torch.zeros((0, 3), device=self.device)

    def solve(self, projections, snap_to_grid=True, voxel_size=1.0):
        """
        Solve the inverse problem: find 3D points that are consistent with
        intersections of backprojected lines from different wire planes.
        
        Args:
            projections (dict): Dictionary mapping plane_id to projection data
            voxel_size (float): Size of the voxels in the volume
            snap_to_grid (bool): Whether to snap intersection points to the nearest grid points
            
        Returns:
            torch.Tensor: 3D points that are consistent with the projections
        """
        if self.debug:
            start_time = time.time()
            print("Solving inverse problem...")
            if snap_to_grid:
                print("  Snapping intersection points to nearest grid points")
        
        # Find intersections between backprojected lines
        intersection_points = self.backproject_planes_to_3d(projections, snap_to_grid, voxel_size)
        
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
