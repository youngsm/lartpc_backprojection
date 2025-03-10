import torch
import numpy as np
import time
import math
from .intersection_solver import LineIntersectionSolver
from tqdm import trange

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
                print("Note: Dense volumes are automatically converted to sparse for efficiency.")
        
        # Convert dense volumes to sparse for consistency and efficiency
        if not isinstance(volume, tuple):
            # Convert to sparse representation
            coords = torch.nonzero(volume, as_tuple=False)
            if coords.shape[0] > 0:
                values = volume[coords[:, 0], coords[:, 1], coords[:, 2]]
            else:
                values = torch.tensor([], device=self.device)
            shape = volume.shape
            
            # Create sparse volume
            sparse_volume = (coords, values, shape)
        else:
            sparse_volume = volume
        
        # Use unified sparse volume projection for all cases
        return self.project_sparse_volume(sparse_volume)
    
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
        from .cuda_kernels import project_sparse_volume
        
        for plane_id, theta in self.solver.plane_angles.items():
            if self.debug:
                plane_start = time.time()
                print(f"  Projecting to plane {plane_id} (theta={theta:.2f})...")
                
            u_min = self.solver.u_min_values[plane_id]
            projection_size = self.solver.projection_sizes[plane_id]
            
            # Use unified sparse projection
            projection = project_sparse_volume(
                coords=coords, 
                values=values, 
                volume_shape=shape,
                theta=theta, 
                u_min=u_min, 
                device=self.device, 
                projection_size=projection_size,
                differentiable=False
            )
            projections[plane_id] = projection
            
            if self.debug:
                plane_end = time.time()
                print(f"    Projection shape: {projection.shape}")
                print(f"    Non-zero elements: {torch.count_nonzero(projection).item()}")
                print(f"    Completed in {plane_end - plane_start:.2f} seconds")
        
        if self.debug:
            end_time = time.time()
            print(f"Projection to {len(projections)} planes completed in {end_time - start_time:.2f} seconds")
        
        return projections
    
    def project_volume_differentiable(self, volume):
        """
        Differentiable version of volume projection that maintains gradients
        for backpropagation through the entire operation.
        
        Args:
            volume (torch.Tensor or tuple): 3D volume of shape (N, N, N) or a sparse volume representation
                                           as (coords, values, shape)
            
        Returns:
            dict: Dictionary mapping plane_id to projection data with gradient tracking
        """
        if self.debug:
            start_time = time.time()
            if isinstance(volume, tuple) and len(volume) == 3:
                coords, values, shape = volume
                print(f"Differentiable projection of sparse volume with {coords.shape[0]} non-zero voxels...")
            else:
                print(f"Differentiable projection of dense volume with shape {volume.shape}...")
                print("Note: Dense volumes are automatically converted to sparse for efficiency.")
        
        # Convert dense volumes to sparse for consistency and efficiency
        if not isinstance(volume, tuple):
            # Convert to sparse representation
            coords = torch.nonzero(volume, as_tuple=False)
            if coords.shape[0] > 0:
                values = volume[coords[:, 0], coords[:, 1], coords[:, 2]]
            else:
                values = torch.tensor([], device=self.device)
            shape = volume.shape
            
            # Create sparse volume
            sparse_volume = (coords, values, shape)
        else:
            sparse_volume = volume
        
        # Use differentiable sparse volume projection
        return self.project_sparse_volume_differentiable(sparse_volume)
    
    def project_sparse_volume_differentiable(self, sparse_volume):
        """
        Differentiable version of sparse volume projection that maintains gradients
        for backpropagation through the entire operation.
        
        Args:
            sparse_volume (tuple): Sparse volume representation as (coords, values, shape)
            
        Returns:
            dict: Dictionary mapping plane_id to projection data with gradient tracking
        """
        coords, values, shape = sparse_volume
        
        if self.debug:
            start_time = time.time()
            print(f"Differentiable projection of sparse volume from {coords.shape[0]} non-zero voxels...")
        
        # Move to the device if needed
        if coords.device.type != self.device:
            coords = coords.to(self.device)
        if values.device.type != self.device:
            values = values.to(self.device)
        
        # Project volume for each plane
        projections = {}
        from .cuda_kernels import project_sparse_volume
        
        for plane_id, theta in self.solver.plane_angles.items():
            if self.debug:
                plane_start = time.time()
                print(f"  Projecting to plane {plane_id} (theta={theta:.2f}) with gradient tracking...")
                
            u_min = self.solver.u_min_values[plane_id]
            projection_size = self.solver.projection_sizes[plane_id]
            
            # Use unified sparse projection with differentiable=True
            projection = project_sparse_volume(
                coords=coords, 
                values=values, 
                volume_shape=shape,
                theta=theta, 
                u_min=u_min, 
                device=self.device, 
                projection_size=projection_size,
                differentiable=True
            )
            projections[plane_id] = projection
            
            if self.debug:
                plane_end = time.time()
                print(f"    Projection shape: {projection.shape}")
                print(f"    Non-zero elements: {torch.count_nonzero(projection).item()}")
                print(f"    Completed in {plane_end - plane_start:.2f} seconds")
        
        if self.debug:
            end_time = time.time()
            print(f"Differentiable projection to {len(projections)} planes completed in {end_time - start_time:.2f} seconds")
        
        return projections
    
    def reconstruct_from_projections(self, projections, threshold=0.1, snap_to_grid=True):
        """
        Reconstruct 3D points from 2D projections.
        
        Args:
            projections (dict): Dictionary mapping plane_id to projection data
            threshold (float): Threshold for considering a hit in the projections
            snap_to_grid (bool): Whether to snap intersection points to the nearest grid points
            
        Returns:
            torch.Tensor: Reconstructed 3D points
        """
        if self.debug:
            start_time = time.time()
            print(f"Reconstructing 3D points from {len(projections)} projections (threshold={threshold})...")
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
        reconstructed_points = self.solver.solve_inverse_problem(thresholded_projections, snap_to_grid=snap_to_grid)
        
        if self.debug:
            end_time = time.time()
            print(f"  Reconstructed {reconstructed_points.shape[0]} points")
            print(f"Total reconstruction time: {end_time - start_time:.2f} seconds")
        
        return reconstructed_points
    
    def reconstruct_volume(self, projections, threshold=0.1, voxel_size=1.0, use_gaussian=True, snap_to_grid=True):
        """
        Reconstruct a 3D volume from 2D projections by placing voxels at intersection points.
        
        Args:
            projections (dict): Dictionary mapping plane_id to projection data
            threshold (float): Threshold for considering a hit in the projections
            voxel_size (float): Size of the voxels to place at intersection points
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
        points = self.reconstruct_from_projections(projections, threshold, snap_to_grid)
        
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
    
    def reconstruct_sparse_volume(self, projections, threshold=0.1, voxel_size=1.0, use_gaussian=False, snap_to_grid=True):
        """
        Reconstruct a sparse 3D volume from 2D projections by placing voxels at intersection points.
        Uses vectorized operations for better performance.
        
        Args:
            projections (dict): Dictionary mapping plane_id to projection data
            threshold (float): Threshold for considering a hit in the projections
            voxel_size (float): Size of the voxels to place at intersection points
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
        points = self.reconstruct_from_projections(projections, threshold, snap_to_grid)
        
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
            unique_values = torch.ones(len(unique_ids), device=self.device)  # All voxels have value 1.0
            
            # VECTORIZED APPROACH - Find the first occurrence of each unique ID
            # Create a tensor marking the first occurrence of each unique value
            first_occurrence = torch.zeros_like(voxel_ids, dtype=torch.bool)
            first_occurrence[torch.unique(inverse_indices, return_inverse=True)[1]] = True
            
            # Extract coordinates of the first occurrences
            unique_coords = indices[first_occurrence]
            
            if self.debug:
                end_time = time.time()
                print(f"  Created sparse volume with {len(unique_ids)} voxels in {end_time - points_time:.2f} seconds")
            
            return (unique_coords, unique_values, self.volume_shape)
        

        # If use_gaussian, the following (slow) code will be used
        
        # Parameters for Gaussian blobs
        sigma = max(1.0, voxel_size)
        radius = int(2 * sigma)
        
        if self.debug:
            indices_time = time.time()
            print(f"  Computed voxel indices in {indices_time - points_time:.2f} seconds")
            print(f"  Creating Gaussian blobs in parallel (radius={radius})...")
        
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
        Evaluate reconstruction quality by comparing the original and reconstructed volumes.
        
        Args:
            original_volume (torch.Tensor): Original 3D volume
            reconstructed_volume (torch.Tensor): Reconstructed 3D volume
            threshold (float): Threshold for binarizing volumes
            
        Returns:
            dict: Dictionary of evaluation metrics
        """
        if self.debug:
            print(f"Evaluating reconstruction quality...")
        
        # Ensure volumes are on the same device
        if original_volume.device.type != self.device:
            original_volume = original_volume.to(self.device)
        if reconstructed_volume.device.type != self.device:
            reconstructed_volume = reconstructed_volume.to(self.device)
        
        # Apply threshold to create binary volumes
        original_binary = (original_volume > threshold).float()
        reconstructed_binary = (reconstructed_volume > threshold).float()
        
        # Calculate metrics
        intersection = torch.sum(original_binary * reconstructed_binary).item()
        union = torch.sum(torch.clamp(original_binary + reconstructed_binary, 0, 1)).item()
        
        # IoU (Intersection over Union)
        iou = intersection / max(1e-8, union)
        
        # Dice coefficient
        dice = 2 * intersection / max(1e-8, torch.sum(original_binary).item() + torch.sum(reconstructed_binary).item())
        
        # Precision and recall
        precision = intersection / max(1e-8, torch.sum(reconstructed_binary).item())
        recall = intersection / max(1e-8, torch.sum(original_binary).item())
        
        # F1 score
        f1 = 2 * precision * recall / max(1e-8, precision + recall)
        
        # MSE
        mse = torch.mean((original_volume - reconstructed_volume) ** 2).item()
        
        # PSNR
        max_val = max(torch.max(original_volume).item(), torch.max(reconstructed_volume).item())
        psnr = 20 * math.log10(max_val / max(1e-8, math.sqrt(mse)))
        
        # PSNR only on non-zero pixels of the target image
        # Create a mask for non-zero pixels in the original volume or reconstructed volume
        non_zero_mask = (reconstructed_volume > 0) | (original_volume > 0)
        
        # If there are any non-zero pixels, compute PSNR only on those
        if torch.any(non_zero_mask):
            # Extract original and reconstructed values at non-zero locations
            original_nonzero = original_volume[non_zero_mask]
            reconstructed_nonzero = reconstructed_volume[non_zero_mask]
            
            # Calculate MSE and PSNR on non-zero regions only
            mse_nonzero = torch.mean((original_nonzero - reconstructed_nonzero) ** 2).item()
            psnr_nonzero = 20 * math.log10(max_val / max(1e-8, math.sqrt(mse_nonzero)))
        else:
            # If no non-zero pixels, set to same as regular PSNR
            psnr_nonzero = psnr
            mse_nonzero = mse
        
        metrics = {
            'iou': iou,
            'dice': dice,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'mse': mse,
            'psnr': psnr,
            'mse_nonzero': mse_nonzero,
            'psnr_nonzero': psnr_nonzero
        }
        
        if self.debug:
            print(f"  IoU: {iou:.4f}")
            print(f"  Dice: {dice:.4f}")
            print(f"  Precision: {precision:.4f}")
            print(f"  Recall: {recall:.4f}")
            print(f"  F1: {f1:.4f}")
            print(f"  MSE: {mse:.6f}")
            print(f"  PSNR: {psnr:.2f} dB")
            print(f"  MSE (non-zero only): {mse_nonzero:.6f}")
            print(f"  PSNR (non-zero only): {psnr_nonzero:.2f} dB")
        
        return metrics
    
    def optimize_sparse_point_intensities(self, candidate_points, target_projections, 
                                         num_iterations=200, lr=0.01, pruning_threshold=0.05, 
                                         pruning_interval=50, l1_weight=0.01):
        """
        Optimize the intensity values of a set of candidate 3D points to match target projections.
        Only optimizes intensity values (alpha) while keeping point positions fixed.
        Periodically prunes low-intensity points.
        
        Args:
            candidate_points (torch.Tensor): Tensor of candidate point coordinates (N, 3)
            target_projections (dict): Dictionary mapping plane_id to target projection data
            num_iterations (int): Number of optimization iterations
            lr (float): Learning rate for optimizer
            pruning_threshold (float): Threshold for pruning low-intensity points
            pruning_interval (int): Interval (in iterations) for pruning points
            l1_weight (float): Weight for L1 regularization
            
        Returns:
            tuple: (optimized_coords, optimized_values, loss_history, num_points_history)
        """
        import torch.optim as optim
        
        if self.debug:
            start_time = time.time()
            print(f"Optimizing intensities for {candidate_points.shape[0]} candidate points...")
            print(f"  Iterations: {num_iterations}, Learning rate: {lr}")
            print(f"  Pruning threshold: {pruning_threshold}, Interval: {pruning_interval}")
        
        # Move candidate points to device if needed
        if candidate_points.device.type != self.device:
            candidate_points = candidate_points.to(self.device)
        
        # Initialize intensity values with small random values
        alpha_values = torch.ones(candidate_points.shape[0], device=self.device)
        alpha_values.requires_grad_(True)  # Only alpha values are trainable
        
        # Create optimizer for alpha values only
        optimizer = optim.Adam([alpha_values], lr=lr)
        
        # Track optimization progress
        loss_values = []
        num_points_history = [candidate_points.shape[0]]
        current_coords = candidate_points.clone()

        # Warmup iterations
        warmup_iterations = int(0.1 * num_iterations)        
        # Optimization loop
        for iteration in trange(num_iterations):
            if self.debug and (iteration == 0 or (iteration + 1) % 20 == 0):
                iter_start = time.time()
                print(f"  Iteration {iteration+1}/{num_iterations}, Points: {current_coords.shape[0]}")
            
            # Create current sparse volume with optimized alpha values
            current_sparse = (current_coords, alpha_values, self.volume_shape)
            
            # Forward pass: project current sparse volume
            current_projections = self.project_sparse_volume_differentiable(current_sparse)
            
            # Calculate loss (MSE between original and current projections)
            loss = 0
            for plane_id in target_projections:
                plane_loss = torch.mean(torch.abs(target_projections[plane_id] - current_projections[plane_id]))
                loss += plane_loss
                
                if self.debug and (iteration == 0 or (iteration + 1) % 20 == 0):
                    print(f"    Plane {plane_id} loss: {plane_loss.item():.6f}")
            
            # Add L1 regularization for sparsity
            l1_reg = l1_weight * torch.mean(torch.abs(alpha_values))
            loss += l1_reg
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Apply constraints: clamp alpha values to [0, 1] range
            with torch.no_grad():
                alpha_values.clamp_(0, 1)
            
            # Store loss
            loss_values.append(loss.item())
            
            # Periodically prune low-alpha points after warmup period
            if iteration >= warmup_iterations and (iteration + 1) % pruning_interval == 0:
                with torch.no_grad():
                    # Find points with alpha values above threshold
                    valid_mask = alpha_values > pruning_threshold
                    
                    # Keep only valid points
                    current_coords = current_coords[valid_mask]
                    new_alpha_values = alpha_values[valid_mask].detach().clone()
                    new_alpha_values.requires_grad_(True)
                    
                    # Replace the optimized parameter
                    del optimizer  # Release the old optimizer
                    alpha_values = new_alpha_values
                    
                    # Reduce learning rate over time
                    new_lr = lr * (0.5 ** (iteration // pruning_interval))
                    optimizer = optim.Adam([alpha_values], lr=new_lr)
                    
                    if self.debug:
                        print(f"    Pruned to {current_coords.shape[0]} points (removed {(~valid_mask).sum().item()} points)")
                        print(f"    New learning rate: {new_lr:.6f}")
                    
                    num_points_history.append(current_coords.shape[0])
            
            if self.debug and (iteration == 0 or (iteration + 1) % 20 == 0):
                iter_end = time.time()
                print(f"    Loss: {loss.item():.6f}, Time: {iter_end - iter_start:.2f}s")
        
        if self.debug:
            end_time = time.time()
            print(f"Optimization completed in {end_time - start_time:.2f} seconds")
            print(f"Final number of points: {current_coords.shape[0]}")
            print(f"Final loss: {loss_values[-1]:.6f}")
        
        return current_coords, alpha_values.detach(), loss_values, num_points_history
    
    def optimize_sparse_point_intensities_admm(self, candidate_points, target_projections, 
                                              num_iterations=100, rho=1.0, alpha=1.5, 
                                              pruning_threshold=0.05, pruning_interval=20,
                                              l1_weight=0.01, relaxation=1.0):
        """
        Optimize the intensity values of a set of candidate 3D points to match target projections
        using the Alternating Direction Method of Multipliers (ADMM). This is an alternative to
        the SGD-based approach, which can handle constraints more effectively.
        
        Args:
            candidate_points (torch.Tensor): Tensor of candidate point coordinates (N, 3)
            target_projections (dict): Dictionary mapping plane_id to target projection data
            num_iterations (int): Number of ADMM iterations
            rho (float): Penalty parameter for ADMM
            alpha (float): Over-relaxation parameter (typically between 1.0 and 1.8)
            pruning_threshold (float): Threshold for pruning low-intensity points
            pruning_interval (int): Interval (in iterations) for pruning points
            l1_weight (float): Weight for L1 regularization term (sparsity)
            relaxation (float): Relaxation parameter for ADMM updates (typically 1.0-1.8)
            
        Returns:
            tuple: (optimized_coords, optimized_values, loss_history, num_points_history)
        """
        if self.debug:
            start_time = time.time()
            print(f"Optimizing intensities for {candidate_points.shape[0]} candidate points using ADMM...")
            print(f"  Iterations: {num_iterations}, Rho: {rho}, Alpha: {alpha}")
            print(f"  Pruning threshold: {pruning_threshold}, Interval: {pruning_interval}")
        
        # Move candidate points to device if needed
        if candidate_points.device.type != self.device:
            candidate_points = candidate_points.to(self.device)
        
        # Initialize intensity values with small random values
        alpha_values = torch.rand(candidate_points.shape[0], device=self.device) * 0.1
        
        # Number of planes (projections)
        num_planes = len(target_projections)
        
        # Create auxiliary variables and dual variables (Lagrangian multipliers)
        # One set for each projection plane
        z_vars = {}
        u_vars = {}
        
        # Initialize auxiliary variables (one for each projection plane)
        for plane_id in target_projections:
            # Create current sparse volume with initial alpha values
            current_sparse = (candidate_points, alpha_values, self.volume_shape)
            
            # Project current sparse volume to get initial projections
            initial_projections = self.project_sparse_volume_differentiable(current_sparse)
            
            # Initialize z as the initial projection
            z_vars[plane_id] = initial_projections[plane_id].clone()
            
            # Initialize dual variables u as zeros with same shape as projection
            u_vars[plane_id] = torch.zeros_like(z_vars[plane_id])
        
        # Track optimization progress
        loss_values = []
        num_points_history = [candidate_points.shape[0]]
        current_coords = candidate_points.clone()
        
        # Warm-up iterations
        warmup_iterations = int(0.1 * num_iterations)
        
        # Optimization loop
        for iteration in trange(num_iterations):
            if self.debug and (iteration == 0 or (iteration + 1) % 10 == 0):
                iter_start = time.time()
                print(f"  Iteration {iteration+1}/{num_iterations}, Points: {current_coords.shape[0]}")
            
            # Step 1: x-update (volume update)
            # This requires solving a least-squares problem with regularization
            
            # Accumulate terms for the x-update
            grad_sum = torch.zeros_like(alpha_values)
            hessian_diag = torch.zeros_like(alpha_values)
            
            # Process each plane
            for plane_id in target_projections:
                # Create current sparse volume with the alpha values
                current_sparse = (current_coords, alpha_values, self.volume_shape)
                
                # Get the projection operator for this plane
                # We need to compute the Jacobian of the projection w.r.t. alpha values
                # We'll do this by computing projections with autograd enabled
                alpha_values.requires_grad_(True)
                
                # Forward projection
                proj = self.project_sparse_volume_differentiable(current_sparse)[plane_id]
                
                # Compute Ax - b + u (residual)
                residual = proj - z_vars[plane_id] + u_vars[plane_id]
                
                # Compute gradient w.r.t. alpha values
                residual_sum = torch.sum(residual)
                residual_sum.backward()
                
                # Extract and accumulate gradient
                if alpha_values.grad is not None:
                    grad_sum += rho * alpha_values.grad
                    
                    # Approximate Hessian diagonal for preconditioned update
                    # This is a simple approximation; a more accurate approach would compute 
                    # the full Hessian or use automatic differentiation for matrix-free operations
                    hessian_diag += rho * (alpha_values.grad ** 2 + 1e-8)
                    
                    # Reset gradient
                    alpha_values.grad.zero_()
                
                alpha_values.requires_grad_(False)
            
            # Add regularization term gradient
            grad_sum += l1_weight * torch.sign(alpha_values)
            hessian_diag += l1_weight
            
            # Preconditioned gradient descent update for alpha values
            step_size = 1.0 / (hessian_diag + 1e-8)  # Avoid division by zero
            alpha_values = alpha_values - step_size * grad_sum
            
            # Project to non-negative values (enforce constraint)
            alpha_values = torch.clamp(alpha_values, 0, 1)
            
            # Step 2: z-update (data fitting step)
            # For each projection plane, update z to minimize the augmented Lagrangian
            for plane_id in target_projections:
                # Create current sparse volume with updated alpha values
                current_sparse = (current_coords, alpha_values, self.volume_shape)
                
                # Get current projection
                current_proj = self.project_sparse_volume_differentiable(current_sparse)[plane_id]
                
                # Over-relaxation step
                current_proj_relaxed = relaxation * current_proj + (1 - relaxation) * z_vars[plane_id]
                
                # Solve for z: min_z ||z - b||^2 + (rho/2)||z - Ax - u||^2
                # Closed-form solution: z = (b + rho*(Ax + u))/(1 + rho)
                target_proj = target_projections[plane_id]
                z_vars[plane_id] = (target_proj + rho * (current_proj_relaxed + u_vars[plane_id])) / (1 + rho)
            
            # Step 3: u-update (dual variable update)
            # For each projection plane, update the dual variables
            primal_residual_norm = 0
            dual_residual_norm = 0
            
            for plane_id in target_projections:
                # Create current sparse volume with updated alpha values
                current_sparse = (current_coords, alpha_values, self.volume_shape)
                
                # Get current projection
                current_proj = self.project_sparse_volume_differentiable(current_sparse)[plane_id]
                
                # Previous z value for calculating dual residual
                prev_z = z_vars[plane_id].clone()
                
                # Dual variable update: u = u + Ax - z
                u_vars[plane_id] = u_vars[plane_id] + current_proj - z_vars[plane_id]
                
                # Calculate residuals for convergence checking
                primal_residual = current_proj - z_vars[plane_id]
                primal_residual_norm += torch.norm(primal_residual).item()
                
                # Dual residual for this plane
                dual_residual = rho * (z_vars[plane_id] - prev_z)
                dual_residual_norm += torch.norm(dual_residual).item()
            
            # Calculate and store loss
            loss = 0
            for plane_id in target_projections:
                # Create current sparse volume with updated alpha values
                current_sparse = (current_coords, alpha_values, self.volume_shape)
                
                # Get current projection
                current_proj = self.project_sparse_volume_differentiable(current_sparse)[plane_id]
                
                # Mean absolute error between projections
                plane_loss = torch.mean(torch.abs(target_projections[plane_id] - current_proj))
                loss += plane_loss.item()
                
                if self.debug and (iteration == 0 or (iteration + 1) % 10 == 0):
                    print(f"    Plane {plane_id} loss: {plane_loss.item():.6f}")
            
            # Add regularization loss
            l1_loss = l1_weight * torch.mean(torch.abs(alpha_values)).item()
            loss += l1_loss
            loss_values.append(loss)
            
            # Adaptive rho update based on residuals
            # This is a common heuristic to adjust rho during ADMM iterations
            if primal_residual_norm > 10 * dual_residual_norm:
                rho *= 2
            elif dual_residual_norm > 10 * primal_residual_norm:
                rho /= 2
                
            # Log convergence metrics
            if self.debug and (iteration == 0 or (iteration + 1) % 10 == 0):
                print(f"    Primal residual: {primal_residual_norm:.6f}, Dual residual: {dual_residual_norm:.6f}")
                print(f"    Updated rho: {rho:.6f}, Total loss: {loss:.6f}")
                print(f"    L1 regularization: {l1_loss:.6f}")
                
            # Periodically prune low-alpha points after warmup period
            if iteration >= warmup_iterations and (iteration + 1) % pruning_interval == 0:
                # Find points with alpha values above threshold
                valid_mask = alpha_values > pruning_threshold
                
                # Keep only valid points
                current_coords = current_coords[valid_mask]
                alpha_values = alpha_values[valid_mask]
                
                # Update number of points history
                num_points_history.append(current_coords.shape[0])
                
                if self.debug:
                    print(f"    Pruned to {current_coords.shape[0]} points")
                    
                # If all points are pruned, break early
                if current_coords.shape[0] == 0:
                    print("Warning: All points were pruned. Consider lowering the pruning threshold.")
                    break
                    
        if self.debug:
            end_time = time.time()
            print(f"ADMM optimization completed in {end_time - start_time:.2f} seconds")
            print(f"Final number of points: {current_coords.shape[0]}")
            print(f"Final loss: {loss_values[-1]:.6f}")
            
        return current_coords, alpha_values, loss_values, num_points_history

    def reconstruct_sparse_volume_admm(self, projections, threshold=0.1, voxel_size=1.0, 
                                      num_iterations=100, rho=1.0, alpha=1.5,
                                      pruning_threshold=0.05, pruning_interval=20,
                                      l1_weight=0.01, fast_merge=True, snap_to_grid=True):
        """
        Reconstruct a sparse volume from projections using the ADMM algorithm.
        
        Args:
            projections (dict): Dictionary mapping plane_id to projection data
            threshold (float): Threshold value for candidate point extraction
            voxel_size (float): Size of voxels for placing candidate points
            num_iterations (int): Number of ADMM iterations
            rho (float): Penalty parameter for ADMM
            alpha (float): Over-relaxation parameter (typically between 1.0 and 1.8)
            pruning_threshold (float): Threshold for pruning low-intensity points
            pruning_interval (int): Interval (in iterations) for pruning points
            l1_weight (float): Weight for L1 regularization (sparsity)
            fast_merge (bool): Whether to use the fast merge algorithm for intersections
            snap_to_grid (bool): Whether to snap intersection points to grid
            
        Returns:
            tuple: (coords, values, volume_shape) representing a sparse volume
                  coords: Coordinates of non-zero voxels (N, 3)
                  values: Values at those coordinates (N)
                  volume_shape: Shape of the full volume (x_size, y_size, z_size)
        """
        if self.debug:
            start_time = time.time()
            print(f"Reconstructing sparse volume from {len(projections)} projections using ADMM...")
            
        # 1. Get candidate points using the existing intersection method
        candidate_points = self.reconstruct_from_projections(
            projections, 
            threshold=threshold, 
            fast_merge=fast_merge,
            snap_to_grid=snap_to_grid
        )
        
        if self.debug:
            print(f"Found {candidate_points.shape[0]} candidate points from intersections")
            
        # 2. Optimize the intensity values using ADMM
        optimized_coords, optimized_values, loss_history, num_points_history = \
            self.optimize_sparse_point_intensities_admm(
                candidate_points,
                projections,
                num_iterations=num_iterations,
                rho=rho,
                alpha=alpha,
                pruning_threshold=pruning_threshold,
                pruning_interval=pruning_interval,
                l1_weight=l1_weight
            )
            
        if self.debug:
            end_time = time.time()
            print(f"Reconstruction completed in {end_time - start_time:.2f} seconds")
            print(f"Final sparse volume has {optimized_coords.shape[0]} non-zero voxels")
            
        # Return sparse volume representation
        return (optimized_coords, optimized_values, self.volume_shape)
    
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