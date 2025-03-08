import torch
import numpy as np
from .intersection_solver import LineIntersectionSolver

class LArTPCReconstructor:
    """
    A high-level class that combines methods for projecting, backprojecting,
    and finding intersections for LArTPC reconstruction.
    """
    def __init__(self, volume_shape, tolerance=1.0, merge_tolerance=1.0, device='cuda'):
        """
        Initialize the reconstructor.
        
        Args:
            volume_shape (tuple): Shape of the 3D volume (N, N, N)
            tolerance (float): Tolerance for intersection testing in mm
            merge_tolerance (float): Tolerance for merging nearby intersections in mm
            device (str): Device to use ('cuda' or 'cpu')
        """
        self.volume_shape = volume_shape
        self.tolerance = tolerance
        self.merge_tolerance = merge_tolerance
        self.device = device
        
        # Create the solver
        self.solver = LineIntersectionSolver(
            volume_shape,
            tolerance,
            merge_tolerance,
            device
        )
    
    def project_volume(self, volume):
        """
        Project a 3D volume to 2D projections for multiple wire planes.
        
        Args:
            volume (torch.Tensor): 3D volume of shape (N, N, N)
            
        Returns:
            dict: Dictionary mapping plane_id to projection data
        """
        # Move volume to the device if needed
        if volume.device.type != self.device:
            volume = volume.to(self.device)
        
        # Project volume for each plane
        projections = {}
        for plane_id, theta in self.solver.plane_angles.items():
            u_min = self.solver.u_min_values[plane_id]
            projection = self.solver.project_volume_cuda(volume, theta, u_min)
            projections[plane_id] = projection
        
        return projections
    
    def reconstruct_from_projections(self, projections, threshold=0.1):
        """
        Reconstruct 3D points from 2D projections.
        
        Args:
            projections (dict): Dictionary mapping plane_id to projection data
            threshold (float): Threshold for considering a hit in the projections
            
        Returns:
            torch.Tensor: Reconstructed 3D points
        """
        # Apply threshold to create binary hit maps
        thresholded_projections = {}
        for plane_id, projection in projections.items():
            # Move projection to the device if needed
            if projection.device.type != self.device:
                projection = projection.to(self.device)
            
            # Apply threshold
            thresholded_projections[plane_id] = (projection > threshold).float()
        
        # Solve the inverse problem
        reconstructed_points = self.solver.solve_inverse_problem(thresholded_projections)
        
        return reconstructed_points
    
    def reconstruct_volume(self, projections, threshold=0.1, voxel_size=1.0):
        """
        Reconstruct a 3D volume from 2D projections by placing voxels at intersection points.
        
        Args:
            projections (dict): Dictionary mapping plane_id to projection data
            threshold (float): Threshold for considering a hit in the projections
            voxel_size (float): Size of the voxels to place at intersection points
            
        Returns:
            torch.Tensor: Reconstructed 3D volume
        """
        # Reconstruct 3D points
        points = self.reconstruct_from_projections(projections, threshold)
        
        # Create an empty volume
        volume = torch.zeros(self.volume_shape, device=self.device)
        
        # If no points were reconstructed, return the empty volume
        if points.size(0) == 0:
            return volume
        
        # Round points to nearest voxel indices
        indices = torch.round(points / voxel_size).long()
        
        # Clamp indices to be within the volume
        indices = torch.clamp(
            indices,
            min=torch.zeros(3, device=self.device).long(),
            max=torch.tensor(self.volume_shape, device=self.device).long() - 1
        )
        
        # Place voxels in the volume
        for idx in indices:
            x, y, z = idx
            
            # Place a Gaussian blob at each point
            sigma = max(1.0, voxel_size)
            for dx in range(-int(2*sigma), int(2*sigma) + 1):
                for dy in range(-int(2*sigma), int(2*sigma) + 1):
                    for dz in range(-int(2*sigma), int(2*sigma) + 1):
                        nx, ny, nz = x + dx, y + dy, z + dz
                        if 0 <= nx < self.volume_shape[0] and 0 <= ny < self.volume_shape[1] and 0 <= nz < self.volume_shape[2]:
                            dist = torch.sqrt(torch.tensor(dx**2 + dy**2 + dz**2, device=self.device))
                            value = torch.exp(-dist/sigma)
                            volume[nx, ny, nz] = max(volume[nx, ny, nz], value)
        
        return volume
    
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
    
    def reconstruct_from_lines(self, lines_by_plane):
        """
        Reconstruct 3D points from line objects directly.
        
        Args:
            lines_by_plane (dict): Dictionary mapping plane_id to lists of Line3D objects
            
        Returns:
            np.ndarray: Reconstructed 3D points
        """
        # Find intersections using CPU
        intersection_points = self.solver.intersect_lines_cpu(lines_by_plane)
        
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