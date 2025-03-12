import torch
import numpy as np
import time
import math
from .cuda_kernels import (
    project_coordinates_to_plane,
    project_coordinates_to_plane_exact,
)
from .intersection_solver import LineIntersectionSolver
from tqdm import trange

class LArTPCReconstructor:
    """
    A high-level class that combines methods for projecting, backprojecting,
    and finding intersections for LArTPC reconstruction.
    """
    def __init__(self, volume_shape, intersection_tolerance=1.0, device='cuda', debug=False, plane_angles=None):
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
        self.intersection_tolerance = intersection_tolerance
        self.device = device
        self.debug = debug
        self.plane_angles = plane_angles
        
        # Create the solver
        self.solver = LineIntersectionSolver(
            volume_shape,
            intersection_tolerance,
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
    
    def project_sparse_volume(self, sparse_volume):
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
        
        for plane_id, theta in self.solver.plane_angles.items():
            if self.debug:
                plane_start = time.time()
                print(f"  Projecting to plane {plane_id} (theta={theta:.2f}) with gradient tracking...")
                
            u_min = self.solver.u_min_values[plane_id]
            projection_size = self.solver.projection_sizes[plane_id]
            
            # Use unified sparse projection with differentiable=True
            projection = project_coordinates_to_plane(
                coords=coords,
                values=values,
                volume_shape=shape,
                theta=theta,
                u_min=u_min,
                device=self.device,
                projection_size=projection_size,
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
    
    def reconstruct_from_projections(self, projections, threshold=0.0, snap_to_grid=True, voxel_size=1.0):
        """
        Reconstruct 3D points from 2D projections. Runs linear intersection solver after an
        initial energy thresholding step.
        
        Args:
            projections (dict): Dictionary mapping plane_id to projection data
            threshold (float): Threshold for considering a hit in the projections
            snap_to_grid (bool): Whether to snap intersection points to the nearest grid points
            voxel_size (float): Size of the voxels in the volume
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
        reconstructed_points = self.solver.solve(thresholded_projections, snap_to_grid=snap_to_grid, voxel_size=voxel_size)
        
        if self.debug:
            end_time = time.time()
            print(f"  Reconstructed {reconstructed_points.shape[0]} points")
            print(f"Total reconstruction time: {end_time - start_time:.2f} seconds")
        
        return reconstructed_points
        
    def reconstruct_coords_from_projections(self, projections, threshold=0.1, voxel_size=1.0, use_gaussian=False, sigma=1.0, snap_to_grid=True):
        """
        Reconstruct a sparse 3D volume from 2D projections by placing voxels at intersection points.

        This is the main function that should be used!
        
        Args:
            projections (dict): Dictionary mapping plane_id to projection data
            threshold (float): Energy threshold applied to given projections for considering a hit
            voxel_size (float): Size of the voxels to place at intersection points
            use_gaussian (bool): Whether to place Gaussian blobs at each point (True) or just single voxels (False)
            sigma (float): Standard deviation of the Gaussian blobs
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
        elif snap_to_grid:
            # Round points to nearest voxel indices
            indices = torch.round(points / voxel_size).long()
        else:
            assert not use_gaussian, "`use_gaussian` can not be true if `snap_to_grid` is False" # !TODO: implement gaussian blobs for continuous coordinates
            if self.debug:
                print(f"  Creating sparse volume with continuous coordinates...")
                            
            # Create values (all 1.0 for now)
            unique_values = torch.ones(points.shape[0], device=self.device)
            
            if self.debug:
                end_time = time.time()
                print(f"  Created sparse continuous representation with {points.shape[0]} points")
                print(f"  Completed in {end_time - start_time:.2f} seconds")

            return (points, unique_values, self.volume_shape)


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
            
            # Find the first occurrence of each unique ID
            first_occurrence = torch.zeros_like(voxel_ids, dtype=torch.bool)
            first_occurrence[torch.unique(inverse_indices, return_inverse=True)[1]] = True
            
            # Extract coordinates of the first occurrences
            unique_coords = indices[first_occurrence]
            
            if self.debug:
                end_time = time.time()
                print(f"  Created sparse volume with {len(unique_ids)} voxels in {end_time - points_time:.2f} seconds")
            
            return (unique_coords, unique_values, self.volume_shape)
        
        # If use_gaussian, the following (slow) code will be used
        if self.debug:
            return self.apply_gaussian_blobs(
                points, indices, sigma, points_time, start_time
            )

        return self.apply_gaussian_blobs(points, indices, sigma)

    
    def optimize_point_intensities(self, candidate_points, target_projections, 
                                         num_iterations=200, lr=0.01, pruning_threshold=0.05, 
                                         pruning_interval=50, l1_weight=0.01, loss_func='l1',
                                         warmup_iterations=500):
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
        alpha_values = torch.rand(candidate_points.shape[0], device=self.device) * 0.1
        alpha_values.requires_grad_(True)  # Only alpha values are trainable
        
        # Create optimizer for alpha values only
        optimizer = optim.Adam([alpha_values], lr=lr)
        
        # Track optimization progress
        loss_values = []
        fidelity_loss_values = []
        l1_loss_values = []
        num_points_history = [candidate_points.shape[0]]
        current_coords = candidate_points.clone()

        # Warmup iterations    
        # Optimization loop
        for iteration in trange(num_iterations):
            if self.debug and (iteration == 0 or (iteration + 1) % 20 == 0):
                iter_start = time.time()
                print(f"  Iteration {iteration+1}/{num_iterations}, Points: {current_coords.shape[0]}")
            
            # Create current sparse volume with optimized alpha values
            current_sparse = (current_coords, alpha_values, self.volume_shape)
            
            # Forward pass: project current sparse volume
            current_projections = self.project_sparse_volume(current_sparse)
            
            # Calculate loss (MSE between original and current projections)
            loss = 0
            for plane_id in target_projections:
                if loss_func == 'l1':
                    plane_loss = torch.mean(torch.abs(target_projections[plane_id] - current_projections[plane_id]))
                elif loss_func == 'l2':
                    plane_loss = torch.mean((target_projections[plane_id] - current_projections[plane_id]) ** 2)
                else:
                    raise ValueError(f"Unknown loss function: {loss}")

                loss += plane_loss
                
                if self.debug and (iteration == 0 or (iteration + 1) % 20 == 0):
                    print(f"    Plane {plane_id} loss: {plane_loss.item():.6f}")
            
            fidelity_loss_values.append(loss.item())
            # Add L1 regularization for sparsity
            l1_reg = l1_weight * torch.mean(torch.abs(alpha_values))
            l1_loss_values.append(l1_reg.item())
            loss += l1_reg
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Apply constraints: clamp alpha values to [0, 1] range
            with torch.no_grad():
                if self.debug:
                    print(f'  alpha values < 0: {alpha_values[alpha_values < 0].shape[0]}, > 1: {alpha_values[alpha_values > 1].shape[0]} out of {alpha_values.shape[0]}')
                alpha_values.clamp_(0, None)
            
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
                    
                    old_state = optimizer.state[valid_mask]
                    # Replace the optimized parameter
                    del optimizer  # Release the old optimizer
                    alpha_values = new_alpha_values
                    
                    # Reduce learning rate over time
                    new_lr = lr * (0.5 ** (iteration // pruning_interval))
                    optimizer = optim.Adam([alpha_values], lr=new_lr)
                    optimizer.state[:] = old_state
                    
                    if self.debug:
                        print(f"    Pruned to {current_coords.shape[0]} points (removed {(~valid_mask).sum().item()} points)")
                    
                    num_points_history.append(current_coords.shape[0])
            
            if self.debug and (iteration == 0 or (iteration + 1) % 20 == 0):
                iter_end = time.time()
                print(f"    Loss: {loss.item():.6f}, Time: {iter_end - iter_start:.2f}s")
        
        if self.debug:
            end_time = time.time()
            print(f"Optimization completed in {end_time - start_time:.2f} seconds")
            print(f"Final number of points: {current_coords.shape[0]}")
            print(f"Final loss: {loss_values[-1]:.6f}")
        
        return current_coords, alpha_values.detach(), [loss_values, fidelity_loss_values, l1_loss_values], num_points_history, 

    def reconstruct_points_from_projections(
        self,
        projections,
        threshold=0.1,
        num_iterations=200,
        lr=0.01,
        pruning_threshold=0.05,
        pruning_interval=50,
        l1_weight=0.01,
        loss_func="l1",
        snap_to_grid=True,
    ):
        """
        Reconstruct 3D points from 2D projections.
        """
        points = self.reconstruct_from_projections(projections, threshold, snap_to_grid=True)

        final_coords, alpha_values, loss_values, num_points_history = self.optimize_point_intensities(points,
                                                                                                      projections,
                                                                                                      num_iterations=num_iterations,
                                                                                                      lr=lr,
                                                                                                      pruning_threshold=pruning_threshold,
                                                                                                      pruning_interval=pruning_interval,
                                                                                                      l1_weight=l1_weight,
                                                                                                      loss_func=loss_func)
        return final_coords, alpha_values

    def evaluate_reconstruction(
        self, original_volume, reconstructed_volume, threshold=0.1
    ):
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
        union = torch.sum(
            torch.clamp(original_binary + reconstructed_binary, 0, 1)
        ).item()

        # IoU (Intersection over Union)
        iou = intersection / max(1e-8, union)

        # Dice coefficient
        dice = (
            2
            * intersection
            / max(
                1e-8,
                torch.sum(original_binary).item()
                + torch.sum(reconstructed_binary).item(),
            )
        )

        # Precision and recall
        precision = intersection / max(1e-8, torch.sum(reconstructed_binary).item())
        recall = intersection / max(1e-8, torch.sum(original_binary).item())

        # F1 score
        f1 = 2 * precision * recall / max(1e-8, precision + recall)

        # MSE
        mse = torch.mean((original_volume - reconstructed_volume) ** 2).item()

        # PSNR
        max_val = max(
            torch.max(original_volume).item(), torch.max(reconstructed_volume).item()
        )
        psnr = 20 * math.log10(max_val / max(1e-8, math.sqrt(mse)))

        # PSNR only on non-zero pixels of the target image
        # Create a mask for non-zero pixels in the original volume or reconstructed volume
        non_zero_mask = (reconstructed_volume > 0) | (original_volume > 0)

        # If there are any non-zero pixels, compute PSNR only on those
        if non_zero_mask.any():
            # Extract original and reconstructed values at non-zero locations
            original_nonzero = original_volume[non_zero_mask]
            reconstructed_nonzero = reconstructed_volume[non_zero_mask]

            # Calculate MSE and PSNR on non-zero regions only
            mse_nonzero = torch.mean(
                (original_nonzero - reconstructed_nonzero) ** 2
            ).item()
            psnr_nonzero = 20 * math.log10(max_val / max(1e-8, math.sqrt(mse_nonzero)))
        else:
            # If no non-zero pixels, set to same as regular PSNR
            psnr_nonzero = np.nan
            mse_nonzero = np.nan

        metrics = {
            "iou": iou,
            "dice": dice,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "mse": mse,
            "psnr": psnr,
            "mse_nonzero": mse_nonzero,
            "psnr_nonzero": psnr_nonzero,
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

    def apply_gaussian_blobs(self, points, indices, sigma, points_time=0, start_time=0):
        # Parameters for Gaussian blobs
        radius = int(2 * sigma)

        if self.debug:
            indices_time = time.time()
            print(
                f"  Computed voxel indices in {indices_time - points_time:.2f} seconds"
            )
            print(f"  Creating Gaussian blobs in parallel (radius={radius})...")

        # First, create offsets for Gaussian blobs
        r = radius
        offsets = []
        for dx in range(-r, r + 1):
            for dy in range(-r, r + 1):
                for dz in range(-r, r + 1):
                    # Filter out offsets that are too far (outside the blob)
                    dist = math.sqrt(dx**2 + dy**2 + dz**2)
                    if (
                        dist <= radius * 1.5
                    ):  # Slightly larger than radius to include all significant values
                        offsets.append((dx, dy, dz, dist))

        # Convert offsets to tensor for vectorized operations
        offset_dists = torch.tensor(
            [(o[0], o[1], o[2]) for o in offsets], device=self.device
        )
        offset_values = torch.tensor(
            [math.exp(-((o[3] / sigma) ** 2) / 2) for o in offsets], device=self.device
        )

        # Only keep offsets where values are significant
        significant_mask = offset_values > 0.01
        offset_dists = offset_dists[significant_mask]
        offset_values = offset_values[significant_mask]

        if self.debug:
            offsets_time = time.time()
            print(
                f"  Created {len(offset_dists)} offset vectors in {offsets_time - indices_time:.2f} seconds"
            )
            print(f"  Generating all voxel coordinates and values...")

        # For each point, add all offsets to create voxel coordinates
        all_coords = []
        all_values = []

        # Process points in batches to avoid memory issues
        batch_size = 1000  # !TODO sy(3.11.25): make this dynamic?
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
            values_batch = (
                offset_values.unsqueeze(0).expand(batch_indices.size(0), -1).reshape(-1)
            )

            # Filter out coordinates outside the volume
            valid_mask = (
                (coords_batch[:, 0] >= 0)
                & (coords_batch[:, 0] < self.volume_shape[0])
                & (coords_batch[:, 1] >= 0)
                & (coords_batch[:, 1] < self.volume_shape[1])
                & (coords_batch[:, 2] >= 0)
                & (coords_batch[:, 2] < self.volume_shape[2])
            )

            coords_batch = coords_batch[valid_mask]
            values_batch = values_batch[valid_mask]

            all_coords.append(coords_batch)
            all_values.append(values_batch)

            if self.debug and (b + 1) % 10 == 0:
                print(
                    f"    Processed batch {b + 1}/{num_batches} ({end_idx}/{points.size(0)} points)..."
                )

        # Concatenate all batches
        if all_coords:
            all_coords = torch.cat(all_coords, dim=0)
            all_values = torch.cat(all_values, dim=0)

            if self.debug:
                concat_time = time.time()
                print(
                    f"  Generated {all_coords.shape[0]} voxel coordinates in {concat_time - offsets_time:.2f} seconds"
                )
                print(f"  Handling duplicate coordinates...")

            # Handle duplicate coordinates using scatter operations instead of loops
            # Convert 3D coordinates to flat indices for a sparse tensor
            flat_indices = (
                all_coords[:, 0] * (self.volume_shape[1] * self.volume_shape[2])
                + all_coords[:, 1] * self.volume_shape[2]
                + all_coords[:, 2]
            )

            # Use PyTorch's scatter_reduce operation to handle duplicates
            # Create a sparse tensor representation
            sparse_dims = torch.prod(torch.tensor(self.volume_shape)).item()
            sparse_tensor = torch.zeros(sparse_dims, device=self.device)

            # For each coordinate, take the maximum value if there are duplicates
            sparse_tensor.scatter_reduce_(0, flat_indices, all_values, reduce="amax")

            # Get non-zero indices and values from the sparse tensor
            non_zero_indices = torch.nonzero(sparse_tensor, as_tuple=False).squeeze()
            non_zero_values = sparse_tensor[non_zero_indices]

            # Convert flat indices back to 3D coordinates
            unique_coords = torch.zeros(
                (non_zero_indices.shape[0], 3), dtype=torch.long, device=self.device
            )
            unique_coords[:, 0] = non_zero_indices // (
                self.volume_shape[1] * self.volume_shape[2]
            )
            unique_coords[:, 1] = (
                non_zero_indices % (self.volume_shape[1] * self.volume_shape[2])
            ) // self.volume_shape[2]
            unique_coords[:, 2] = non_zero_indices % self.volume_shape[2]

            if self.debug:
                unique_time = time.time()
                print(
                    f"  Final sparse volume has {unique_coords.shape[0]} non-zero voxels"
                )
                print(
                    f"  Duplicate handling completed in {unique_time - concat_time:.2f} seconds"
                )
                print(
                    f"Total sparse volume reconstruction time: {unique_time - start_time:.2f} seconds"
                )

            return (unique_coords, non_zero_values, self.volume_shape)
        else:
            # Return empty sparse volume
            if self.debug:
                end_time = time.time()
                print("  No non-zero values in sparse volume")
                print(
                    f"Total sparse volume reconstruction time: {end_time - start_time:.2f} seconds"
                )

            return (
                torch.zeros((0, 3), dtype=torch.long, device=self.device),
                torch.zeros(0, device=self.device),
                self.volume_shape,
            )