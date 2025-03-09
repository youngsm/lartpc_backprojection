import torch
import numpy as np
import time
import math
from .intersection_solver import LineIntersectionSolver

class LArTPCReconstructor:
    """
    A high-level class that combines methods for projecting, backprojecting,
    and finding intersections for LArTPC reconstruction.
    """
    def __init__(self, volume_shape, tolerance=1.0, merge_tolerance=1.0, device='cuda', debug=False, plane_angles=None):
        """
        Initialize the reconstructor.
        
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
        self.plane_angles = plane_angles
        
        # Create the solver
        self.solver = LineIntersectionSolver(
            volume_shape,
            tolerance,
            merge_tolerance,
            device,
            debug=debug,
            plane_angles=plane_angles
        )
    
    def set_plane_angles(self, plane_angles):
        """
        Update the plane angles used for projection and backprojection.
        
        Args:
            plane_angles (dict): Dictionary mapping plane_id to angle in radians
        """
        self.plane_angles = plane_angles
        self.solver.set_plane_angles(plane_angles)
    
    def project_volume(self, volume):
        """
        Project a 3D volume to 2D projections for multiple wire planes.
        
        Args:
            volume (torch.Tensor or tuple): 3D volume of shape (N, N, N) or a sparse volume representation
                                           as (coords, values, shape)
            
        Returns:
            dict: Dictionary mapping plane_id to projection data
        """
        if self.debug:
            start_time = time.time()
            if isinstance(volume, tuple) and len(volume) == 3:
                coords, values, shape = volume
                print(f"Projecting sparse volume with {coords.shape[0]} non-zero voxels...")
            else:
                print(f"Projecting dense volume with shape {volume.shape}...")
        
        # Check if volume is a sparse representation
        if isinstance(volume, tuple) and len(volume) == 3:
            result = self.project_sparse_volume(volume)
        else:
            # Move volume to the device if needed
            if volume.device.type != self.device:
                volume = volume.to(self.device)
            
            # Project volume for each plane
            result = {}
            for plane_id, theta in self.solver.plane_angles.items():
                u_min = self.solver.u_min_values[plane_id]
                result[plane_id] = self.solver.project_volume_cuda(volume, theta, u_min)
        
        if self.debug:
            end_time = time.time()
            print(f"Projection to {len(result)} planes completed in {end_time - start_time:.2f} seconds")
        
        return result
    
    def project_sparse_volume(self, sparse_volume):
        """
        Project a sparse 3D volume to 2D projections for multiple wire planes.
        
        Args:
            sparse_volume (tuple): Sparse volume representation as (coords, values, shape)
            
        Returns:
            dict: Dictionary mapping plane_id to projection data
        """
        coords, values, shape = sparse_volume
        
        if self.debug:
            start_time = time.time()
            print(f"Projecting sparse volume from {coords.shape[0]} non-zero voxels...")
        
        # Move to the device if needed
        if coords.device.type != self.device:
            coords = coords.to(self.device)
        if values.device.type != self.device:
            values = values.to(self.device)
        
        # Project volume for each plane
        projections = {}
        for plane_id, theta in self.solver.plane_angles.items():
            if self.debug:
                plane_start = time.time()
                print(f"  Projecting to plane {plane_id} (theta={theta:.2f})...")
                
            u_min = self.solver.u_min_values[plane_id]
            
            # Use sparse projection directly
            from .cuda_kernels import project_volume_cuda_sparse
            projection = project_volume_cuda_sparse(coords, values, shape, theta, u_min, self.device)
            projections[plane_id] = projection
            
            if self.debug:
                plane_end = time.time()
                print(f"    Projection shape: {projection.shape}")
                print(f"    Non-zero elements: {torch.count_nonzero(projection).item()}")
                print(f"    Completed in {plane_end - plane_start:.2f} seconds")
        
        if self.debug:
            end_time = time.time()
            print(f"All sparse projections completed in {end_time - start_time:.2f} seconds")
        
        return projections
    
    def reconstruct_from_projections(self, projections, threshold=0.1, fast_merge=True, snap_to_grid=True):
        """
        Reconstruct 3D points from 2D projections.
        
        Args:
            projections (dict): Dictionary mapping plane_id to projection data
            threshold (float): Threshold for considering a hit in the projections
            fast_merge (bool): Whether to use fast merge mode (much faster but slightly less precise)
            snap_to_grid (bool): Whether to snap intersection points to the nearest grid points
            
        Returns:
            torch.Tensor: Reconstructed 3D points
        """
        if self.debug:
            start_time = time.time()
            print(f"Reconstructing 3D points from {len(projections)} projections (threshold={threshold})...")
            if fast_merge:
                print("Using FAST merge mode for maximum speed")
            else:
                print("Using PRECISE merge mode (slower but more accurate)")
            if snap_to_grid:
                print("Snapping intersection points to nearest grid points")
        
        # Apply threshold to create binary hit maps
        thresholded_projections = {}
        
        for plane_id, projection in projections.items():
            # Move projection to the device if needed
            if projection.device.type != self.device:
                projection = projection.to(self.device)
            
            if self.debug:
                before_count = torch.count_nonzero(projection).item()
            
            # Apply threshold
            thresholded_projections[plane_id] = (projection > threshold).float()
            
            if self.debug:
                after_count = torch.count_nonzero(thresholded_projections[plane_id]).item()
                print(f"  Plane {plane_id}: {before_count} hits before threshold, {after_count} after")
        
        if self.debug:
            threshold_time = time.time()
            print(f"  Applied thresholds in {threshold_time - start_time:.2f} seconds")
            print(f"  Solving inverse problem...")
        
        # Solve the inverse problem
        reconstructed_points = self.solver.solve_inverse_problem(thresholded_projections, fast_merge=fast_merge, snap_to_grid=snap_to_grid)
        
        if self.debug:
            end_time = time.time()
            print(f"  Reconstructed {reconstructed_points.shape[0]} points")
            print(f"Total reconstruction time: {end_time - start_time:.2f} seconds")
        
        return reconstructed_points
    
    def reconstruct_volume(self, projections, threshold=0.1, voxel_size=1.0, fast_merge=True, use_gaussian=True, snap_to_grid=True):
        """
        Reconstruct a 3D volume from 2D projections by placing voxels at intersection points.
        
        Args:
            projections (dict): Dictionary mapping plane_id to projection data
            threshold (float): Threshold for considering a hit in the projections
            voxel_size (float): Size of the voxels to place at intersection points
            fast_merge (bool): Whether to use fast merge mode for clustering
            use_gaussian (bool): Whether to place Gaussian blobs at each point (True) or just single voxels (False)
            snap_to_grid (bool): Whether to snap intersection points to the nearest grid points
            
        Returns:
            torch.Tensor: Reconstructed 3D volume
        """
        if self.debug:
            start_time = time.time()
            print(f"Reconstructing 3D volume from projections...")
            print(f"  Using {'Gaussian blobs' if use_gaussian else 'single voxels'} for reconstruction")
            if snap_to_grid:
                print(f"  Snapping intersection points to nearest grid points")
        
        # Reconstruct 3D points
        points = self.reconstruct_from_projections(projections, threshold, fast_merge, snap_to_grid)
        
        if self.debug:
            points_time = time.time()
            print(f"  Reconstructed {points.shape[0]} 3D points in {points_time - start_time:.2f} seconds")
            print(f"  Creating volume with voxel size {voxel_size}...")
        
        # Create an empty volume
        volume = torch.zeros(self.volume_shape, device=self.device)
        
        # If no points were reconstructed, return the empty volume
        if points.size(0) == 0:
            if self.debug:
                print("  No points reconstructed, returning empty volume")
            return volume
        
        # Round points to nearest voxel indices if not already snapped to grid
        if snap_to_grid and voxel_size == 1.0:
            # Points are already on the grid, just convert to long
            indices = points.long()
        else:
            # Round points to nearest voxel indices
            indices = torch.round(points / voxel_size).long()
        
        # Clamp indices to be within the volume
        indices = torch.clamp(
            indices,
            min=torch.zeros(3, device=self.device).long(),
            max=torch.tensor(self.volume_shape, device=self.device).long() - 1
        )
        
        if self.debug:
            indices_time = time.time()
            print(f"  Computed voxel indices in {indices_time - points_time:.2f} seconds")
        
        if use_gaussian:
            # Place Gaussian blobs at each point
            if self.debug:
                print(f"  Creating point map for Gaussian blobs...")
                
            # First, create a map of all points in a tensor
            point_map = torch.zeros((self.volume_shape[0], self.volume_shape[1], self.volume_shape[2]), 
                                   dtype=torch.bool, device=self.device)
            
            # Mark locations of points
            if indices.shape[0] > 0:
                point_map[indices[:, 0], indices[:, 1], indices[:, 2]] = True
            
            # Find indices of all points
            point_indices = torch.nonzero(point_map, as_tuple=True)
            
            if self.debug:
                map_time = time.time()
                print(f"  Created point map with {len(point_indices[0])} unique locations in {map_time - indices_time:.2f} seconds")
                print(f"  Placing Gaussian blobs...")
            
            # For each marked point, place a Gaussian blob
            sigma = max(1.0, voxel_size)
            radius = int(2 * sigma)
            
            # Create a Gaussian kernel for efficiency
            kernel_size = 2 * radius + 1
            kernel_1d = torch.exp(-torch.arange(-radius, radius + 1, device=self.device)**2 / (2 * sigma**2))
            # Normalize kernel
            kernel_1d = kernel_1d / kernel_1d.sum()
            
            # Use separable 3D convolution for efficiency
            for i in range(len(point_indices[0])):
                x, y, z = point_indices[0][i], point_indices[1][i], point_indices[2][i]
                
                # Calculate ranges for the Gaussian blob with bounds checking
                x_min, x_max = max(0, x - radius), min(self.volume_shape[0] - 1, x + radius)
                y_min, y_max = max(0, y - radius), min(self.volume_shape[1] - 1, y + radius)
                z_min, z_max = max(0, z - radius), min(self.volume_shape[2] - 1, z + radius)
                
                # Calculate kernel indices
                kx_min, kx_max = radius - (x - x_min), radius + (x_max - x)
                ky_min, ky_max = radius - (y - y_min), radius + (y_max - y)
                kz_min, kz_max = radius - (z - z_min), radius + (z_max - z)
                
                # Get kernel values
                kx = kernel_1d[kx_min:kx_max+1]
                ky = kernel_1d[ky_min:ky_max+1]
                kz = kernel_1d[kz_min:kz_max+1]
                
                # Calculate 3D Gaussian values using outer products
                for dx, kx_val in enumerate(kx):
                    for dy, ky_val in enumerate(ky):
                        for dz, kz_val in enumerate(kz):
                            nx, ny, nz = x_min + dx, y_min + dy, z_min + dz
                            value = kx_val * ky_val * kz_val
                            volume[nx, ny, nz] = max(volume[nx, ny, nz], value)
            
            if self.debug:
                blob_time = time.time()
                print(f"  Placed Gaussian blobs in {blob_time - map_time:.2f} seconds")
        else:
            # Just place single voxels at each point (no Gaussian blobs)
            if self.debug:
                print(f"  Placing single voxels at {indices.shape[0]} points...")
            
            # Set all voxels to 1.0 (or you could use a different value if needed)
            volume[indices[:, 0], indices[:, 1], indices[:, 2]] = 1.0
            
            if self.debug:
                voxel_time = time.time()
                print(f"  Placed single voxels in {voxel_time - indices_time:.2f} seconds")
        
        if self.debug:
            end_time = time.time()
            print(f"  Non-zero voxels in reconstructed volume: {torch.count_nonzero(volume).item()}")
            print(f"Total volume reconstruction time: {end_time - start_time:.2f} seconds")
        
        return volume
    
    def reconstruct_sparse_volume(self, projections, threshold=0.1, voxel_size=1.0, fast_merge=True, use_gaussian=True, snap_to_grid=True):
        """
        Reconstruct a sparse 3D volume from 2D projections by placing voxels at intersection points.
        Uses vectorized operations for better performance.
        
        Args:
            projections (dict): Dictionary mapping plane_id to projection data
            threshold (float): Threshold for considering a hit in the projections
            voxel_size (float): Size of the voxels to place at intersection points
            fast_merge (bool): Whether to use fast merge mode for clustering
            use_gaussian (bool): Whether to place Gaussian blobs at each point (True) or just single voxels (False)
            snap_to_grid (bool): Whether to snap intersection points to the nearest grid points
            
        Returns:
            tuple: (coords, values, shape) representing the sparse reconstructed volume
        """
        if self.debug:
            start_time = time.time()
            print(f"Reconstructing sparse 3D volume from projections...")
            print(f"  Using {'Gaussian blobs' if use_gaussian else 'single voxels'} for reconstruction")
            if snap_to_grid:
                print(f"  Snapping intersection points to nearest grid points")
        
        # Reconstruct 3D points
        points = self.reconstruct_from_projections(projections, threshold, fast_merge, snap_to_grid)
        
        if self.debug:
            points_time = time.time()
            print(f"  Reconstructed {points.shape[0]} 3D points in {points_time - start_time:.2f} seconds")
            
        # If no points were reconstructed, return an empty sparse volume
        if points.size(0) == 0:
            if self.debug:
                print("  No points reconstructed, returning empty sparse volume")
            return (
                torch.zeros((0, 3), dtype=torch.long, device=self.device),
                torch.zeros(0, device=self.device),
                self.volume_shape
            )
        
        if self.debug:
            print(f"  Creating sparse volume with voxel size {voxel_size}...")
        
        # Round points to nearest voxel indices if not already snapped to grid
        if snap_to_grid and voxel_size == 1.0:
            # Points are already on the grid, just convert to long
            indices = points.long()
        else:
            # Round points to nearest voxel indices
            indices = torch.round(points / voxel_size).long()
        
        # Clamp indices to be within the volume
        indices = torch.clamp(
            indices,
            min=torch.zeros(3, device=self.device).long(),
            max=torch.tensor(self.volume_shape, device=self.device).long() - 1
        )
        
        if self.debug:
            indices_time = time.time()
            print(f"  Computed voxel indices in {indices_time - points_time:.2f} seconds")
        
        if not use_gaussian:
            # Just use the voxel indices directly without Gaussian blobs
            if self.debug:
                indices_time = time.time()
                print(f"  Using direct voxel indices without Gaussian blobs")
                
            # Create a unique identifier for each voxel to remove duplicates
            voxel_ids = (
                indices[:, 0] * self.volume_shape[1] * self.volume_shape[2] +
                indices[:, 1] * self.volume_shape[2] +
                indices[:, 2]
            )
            
            # Get unique voxels
            unique_ids, inverse_indices = torch.unique(voxel_ids, return_inverse=True)
            
            # Create sparse volume representation
            unique_coords = torch.zeros((len(unique_ids), 3), dtype=torch.long, device=self.device)
            unique_values = torch.ones(len(unique_ids), device=self.device)  # All voxels have value 1.0
            
            # Get coordinates for each unique voxel
            for i, voxel_id in enumerate(unique_ids):
                mask = (voxel_ids == voxel_id)
                unique_coords[i] = indices[mask][0]  # Take first occurrence
            
            if self.debug:
                end_time = time.time()
                print(f"  Created sparse volume with {len(unique_ids)} voxels in {end_time - points_time:.2f} seconds")
            
            return (unique_coords, unique_values, self.volume_shape)
        
        # Parameters for Gaussian blobs
        sigma = max(1.0, voxel_size)
        radius = int(2 * sigma)
        
        if self.debug:
            indices_time = time.time()
            print(f"  Computed voxel indices in {indices_time - points_time:.2f} seconds")
            print(f"  Creating Gaussian blobs in parallel (radius={radius})...")
        
        # Instead of looping through points, use a vectorized approach to create all potential voxel coordinates
        # First, create offsets for Gaussian blobs
        r = radius
        offsets = []
        for dx in range(-r, r+1):
            for dy in range(-r, r+1):
                for dz in range(-r, r+1):
                    # Filter out offsets that are too far (outside the blob)
                    dist = math.sqrt(dx**2 + dy**2 + dz**2)
                    if dist <= radius * 1.5:  # Slightly larger than radius to include all significant values
                        offsets.append((dx, dy, dz, dist))
        
        # Convert offsets to tensor for vectorized operations
        offset_dists = torch.tensor([(o[0], o[1], o[2]) for o in offsets], device=self.device)
        offset_values = torch.tensor([math.exp(-(o[3]/sigma)**2 / 2) for o in offsets], device=self.device)
        
        # Only keep offsets where values are significant
        significant_mask = offset_values > 0.01
        offset_dists = offset_dists[significant_mask]
        offset_values = offset_values[significant_mask]
        
        if self.debug:
            offsets_time = time.time()
            print(f"  Created {len(offset_dists)} offset vectors in {offsets_time - indices_time:.2f} seconds")
            print(f"  Generating all voxel coordinates and values...")
        
        # For each point, add all offsets to create voxel coordinates
        all_coords = []
        all_values = []
        
        # Process points in batches to avoid memory issues
        batch_size = 1000  # Adjust based on available memory
        num_batches = (points.size(0) + batch_size - 1) // batch_size
        
        for b in range(num_batches):
            start_idx = b * batch_size
            end_idx = min((b + 1) * batch_size, points.size(0))
            batch_indices = indices[start_idx:end_idx]
            
            # Expand dimensions for broadcasting
            batch_points = batch_indices.unsqueeze(1)  # [batch_size, 1, 3]
            offsets_expanded = offset_dists.unsqueeze(0)  # [1, num_offsets, 3]
            
            # Generate all coordinates by adding offsets to each point
            # This creates a tensor of shape [batch_size, num_offsets, 3]
            coords_batch = batch_points + offsets_expanded
            
            # Reshape to [batch_size * num_offsets, 3]
            coords_batch = coords_batch.reshape(-1, 3)
            
            # Generate values for each coordinate
            # Each point contributes to num_offsets voxels with the corresponding offset values
            values_batch = offset_values.unsqueeze(0).expand(batch_indices.size(0), -1).reshape(-1)
            
            # Filter out coordinates outside the volume
            valid_mask = (
                (coords_batch[:, 0] >= 0) & (coords_batch[:, 0] < self.volume_shape[0]) &
                (coords_batch[:, 1] >= 0) & (coords_batch[:, 1] < self.volume_shape[1]) &
                (coords_batch[:, 2] >= 0) & (coords_batch[:, 2] < self.volume_shape[2])
            )
            
            coords_batch = coords_batch[valid_mask]
            values_batch = values_batch[valid_mask]
            
            all_coords.append(coords_batch)
            all_values.append(values_batch)
            
            if self.debug and (b + 1) % 10 == 0:
                print(f"    Processed batch {b+1}/{num_batches} ({end_idx}/{points.size(0)} points)...")
        
        # Concatenate all batches
        if all_coords:
            all_coords = torch.cat(all_coords, dim=0)
            all_values = torch.cat(all_values, dim=0)
            
            if self.debug:
                concat_time = time.time()
                print(f"  Generated {all_coords.shape[0]} voxel coordinates in {concat_time - offsets_time:.2f} seconds")
                print(f"  Handling duplicate coordinates...")
            
            # Handle duplicate coordinates using scatter operations instead of loops
            # Convert 3D coordinates to flat indices for a sparse tensor
            flat_indices = (
                all_coords[:, 0] * (self.volume_shape[1] * self.volume_shape[2]) +
                all_coords[:, 1] * self.volume_shape[2] +
                all_coords[:, 2]
            )
            
            # Use PyTorch's scatter_reduce operation to handle duplicates
            # Create a sparse tensor representation
            sparse_dims = torch.prod(torch.tensor(self.volume_shape)).item()
            sparse_tensor = torch.zeros(sparse_dims, device=self.device)
            
            # For each coordinate, take the maximum value if there are duplicates
            sparse_tensor.scatter_reduce_(0, flat_indices, all_values, reduce='amax')
            
            # Get non-zero indices and values from the sparse tensor
            non_zero_indices = torch.nonzero(sparse_tensor, as_tuple=False).squeeze()
            non_zero_values = sparse_tensor[non_zero_indices]
            
            # Convert flat indices back to 3D coordinates
            unique_coords = torch.zeros((non_zero_indices.shape[0], 3), dtype=torch.long, device=self.device)
            unique_coords[:, 0] = non_zero_indices // (self.volume_shape[1] * self.volume_shape[2])
            unique_coords[:, 1] = (non_zero_indices % (self.volume_shape[1] * self.volume_shape[2])) // self.volume_shape[2]
            unique_coords[:, 2] = non_zero_indices % self.volume_shape[2]
            
            if self.debug:
                unique_time = time.time()
                print(f"  Final sparse volume has {unique_coords.shape[0]} non-zero voxels")
                print(f"  Duplicate handling completed in {unique_time - concat_time:.2f} seconds")
                print(f"Total sparse volume reconstruction time: {unique_time - start_time:.2f} seconds")
            
            return (unique_coords, non_zero_values, self.volume_shape)
        else:
            # Return empty sparse volume
            if self.debug:
                end_time = time.time()
                print("  No non-zero values in sparse volume")
                print(f"Total sparse volume reconstruction time: {end_time - start_time:.2f} seconds")
                
            return (
                torch.zeros((0, 3), dtype=torch.long, device=self.device),
                torch.zeros(0, device=self.device),
                self.volume_shape
            )
    
    def evaluate_reconstruction(self, original_volume, reconstructed_volume, threshold=0.1):
        """
        Evaluate the quality of a reconstruction by comparing it to the original volume.
        
        Args:
            original_volume (torch.Tensor): Original 3D volume
            reconstructed_volume (torch.Tensor): Reconstructed 3D volume
            threshold (float): Threshold for considering a voxel as occupied
            
        Returns:
            dict: Dictionary with evaluation metrics
        """
        # Move volumes to the device if needed
        if original_volume.device.type != self.device:
            original_volume = original_volume.to(self.device)
        if reconstructed_volume.device.type != self.device:
            reconstructed_volume = reconstructed_volume.to(self.device)
        
        # Create binary volumes
        original_binary = original_volume > threshold
        reconstructed_binary = reconstructed_volume > threshold
        
        # Compute intersection
        intersection = torch.logical_and(original_binary, reconstructed_binary)
        
        # Compute union
        union = torch.logical_or(original_binary, reconstructed_binary)
        
        # Compute metrics
        num_original = torch.sum(original_binary).item()
        num_reconstructed = torch.sum(reconstructed_binary).item()
        num_intersection = torch.sum(intersection).item()
        num_union = torch.sum(union).item()
        
        # Compute precision, recall, and IoU
        precision = num_intersection / max(1, num_reconstructed)
        recall = num_intersection / max(1, num_original)
        iou = num_intersection / max(1, num_union)
        f1 = 2 * precision * recall / max(1e-10, precision + recall)
        
        return {
            'precision': precision,
            'recall': recall,
            'iou': iou,
            'f1': f1,
            'num_original': num_original,
            'num_reconstructed': num_reconstructed,
            'num_intersection': num_intersection
        }
    
    def reconstruct_from_lines(self, lines_by_plane, snap_to_grid=True):
        """
        Reconstruct 3D points from line objects directly.
        
        Args:
            lines_by_plane (dict): Dictionary mapping plane_id to lists of Line3D objects
            snap_to_grid (bool): Whether to snap intersection points to the nearest grid points
            
        Returns:
            np.ndarray: Reconstructed 3D points
        """
        # Find intersections using CPU
        intersection_points = self.solver.intersect_lines_cpu(lines_by_plane, snap_to_grid)
        
        return intersection_points
    
    def get_line_directions(self):
        """
        Get the direction vectors for each wire plane.
        
        Returns:
            dict: Dictionary mapping plane_id to direction vectors
        """
        directions = {}
        for plane_id, theta in self.solver.plane_angles.items():
            # Direction perpendicular to the wire (along which the backprojection "smears")
            directions[plane_id] = np.array([
                0.0,                  # Perpendicular to x (the drift direction)
                np.cos(theta),        # y-component
                np.sin(theta)         # z-component
            ])
        
        return directions 