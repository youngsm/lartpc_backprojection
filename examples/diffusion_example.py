#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Example demonstrating electron diffusion effects in LArTPC reconstruction.

This script shows how to enable and configure electron diffusion and attenuation
in the LArTPC reconstruction process. It creates a set of test data points and
shows reconstructions with and without diffusion effects for comparison.
"""

import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
import torch
from matplotlib.colors import LogNorm

# Add the parent directory to the path to import the package
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from lartpc_reconstruction import LArTPCReconstructor
from lartpc_reconstruction.visualization import (
    visualize_volume, 
    visualize_projections, 
    visualize_original_vs_reconstructed
)

# Import the helper functions from the realistic reconstruction example
from realistic_sparse_reconstruction import (
    create_helix,
    create_sine_wave,
    create_volume_from_points
)

def create_particle_track_data(volume_shape=(100, 100, 100), device="cuda"):
    """
    Create realistic particle track data for LArTPC simulation.
    
    This creates a set of tracks that start from the right side of the detector
    (maximum X) and travel toward the left (X=0) anode where they are detected.
    
    Args:
        volume_shape (tuple): Shape of the volume
        device (str): Device to use for computation
        
    Returns:
        tuple: (sparse_volume, dense_volume, points)
    """
    # Create various particle tracks
    input_points = []
    
    # Diagonal track from max_x to min_x (drift direction)
    # This track will show increasing diffusion as it gets closer to X=0
    max_x, max_y, max_z = [v-1 for v in volume_shape]
    track1 = np.stack([
        np.linspace(max_x, 5, 50),      # X from max to near anode
        np.linspace(max_y * 0.2, max_y * 0.8, 50),  # Y increasing
        np.linspace(max_z * 0.2, max_z * 0.8, 50)   # Z increasing
    ], axis=1)
    
    # Horizontal track (constant Y, Z) at different X positions
    # This will show different diffusion amounts for each track
    for x_start in [max_x, max_x * 0.75, max_x * 0.5, max_x * 0.25]:
        track = np.stack([
            np.linspace(x_start, 5, 30),  # X from position to near anode
            np.ones(30) * max_y * 0.5,    # Constant Y
            np.ones(30) * max_z * 0.5     # Constant Z
        ], axis=1)
        input_points.append(track)
    
    # Create a helix from max_x toward anode
    helix_points = create_helix(
        start=(max_x, max_y * 0.25, max_z * 0.25), 
        radius=10, 
        height=max_x - 10,  # Going from max_x toward anode
        turns=4, 
        num_points=100
    )
    
    # Reverse the helix so it starts from max_x
    helix_points = helix_points[::-1]
    
    # Add the tracks to input points
    input_points.append(track1)
    input_points.append(helix_points)
    
    # Combine all points and convert to tensor
    all_points = np.vstack(input_points)
    points_tensor = torch.tensor(all_points, dtype=torch.float32)
    
    # Create volume from points
    sparse_volume, dense_volume, points = create_volume_from_points(
        points_tensor, volume_shape, radius=0.0, device=device
    )
    
    return sparse_volume, dense_volume, points

def compare_diffusion_effects(volume_shape=(100, 100, 100), 
                             diffusion_sigma_t=2.0, 
                             diffusion_sigma_l=1.0,
                             attenuation_coeff=0.01):
    """
    Compare reconstructions with and without electron diffusion effects.
    
    Args:
        volume_shape (tuple): Shape of the 3D volume
        diffusion_sigma_t (float): Transverse diffusion coefficient
        diffusion_sigma_l (float): Longitudinal diffusion coefficient
        attenuation_coeff (float): Electron attenuation coefficient
    """
    print("Comparing LArTPC reconstruction with and without electron diffusion...")
    
    # Use CUDA if available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Create img directory if it doesn't exist
    os.makedirs('img', exist_ok=True)
    
    # Create particle track data
    print("Creating particle track data...")
    ground_truth_sparse, ground_truth, original_points = create_particle_track_data(
        volume_shape, device
    )
    print(f"Created ground truth with {original_points.shape[0]} non-zero points")
    
    # Create reconstructor with three planes
    print("Creating reconstructor with 3 planes...")
    # Standard LArTPC planes at 0°, 60°, and 120° (TPC, U, V)
    plane_angles = {0: 0.0, 1: np.pi/3, 2: 2*np.pi/3}
    
    # Create reconstructor without diffusion
    reconstructor_no_diffusion = LArTPCReconstructor(
        volume_shape=volume_shape, 
        device=device,
        plane_angles=plane_angles,
        debug=True,
        enable_diffusion=False
    )
    
    # Create reconstructor with diffusion
    reconstructor_with_diffusion = LArTPCReconstructor(
        volume_shape=volume_shape, 
        device=device,
        plane_angles=plane_angles,
        debug=True,
        enable_diffusion=True,
        diffusion_sigma_t=diffusion_sigma_t,
        diffusion_sigma_l=diffusion_sigma_l,
        attenuation_coeff=attenuation_coeff
    )
    
    # Visualize ground truth
    print("Visualizing ground truth...")
    fig = visualize_volume(ground_truth.cpu().numpy(), threshold=0.0)
    fig.savefig("img/diffusion_ground_truth.png")
    plt.close(fig)
    
    # Generate projections without diffusion
    print("\nGenerating projections without diffusion...")
    projections_no_diffusion = reconstructor_no_diffusion.project_sparse_volume_differentiable(ground_truth_sparse)
    
    # Visualize projections without diffusion
    fig = visualize_projections({k: v.cpu().detach().numpy() for k, v in projections_no_diffusion.items()})
    fig.suptitle("Projections without Diffusion")
    fig.savefig("img/diffusion_projections_no_diffusion.png")
    plt.close(fig)
    
    # Generate projections with diffusion
    print("\nGenerating projections with diffusion...")
    projections_with_diffusion = reconstructor_with_diffusion.project_sparse_volume_differentiable(ground_truth_sparse)
    
    # Visualize projections with diffusion
    fig = visualize_projections({k: v.cpu().detach().numpy() for k, v in projections_with_diffusion.items()})
    fig.suptitle(f"Projections with Diffusion (σ_t={diffusion_sigma_t}, σ_l={diffusion_sigma_l}, atten={attenuation_coeff})")
    fig.savefig("img/diffusion_projections_with_diffusion.png")
    plt.close(fig)
    
    # Compare projections directly for each plane
    for plane_id in projections_no_diffusion:
        proj_no_diff = projections_no_diffusion[plane_id].cpu().detach().numpy()
        proj_with_diff = projections_with_diffusion[plane_id].cpu().detach().numpy()
        
        # Create side-by-side comparison
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # Original projection (no diffusion)
        im0 = axes[0].imshow(proj_no_diff, aspect='auto', cmap='viridis')
        axes[0].set_title(f"Original (Plane {plane_id})")
        plt.colorbar(im0, ax=axes[0])
        
        # Diffused projection
        im1 = axes[1].imshow(proj_with_diff, aspect='auto', cmap='viridis')
        axes[1].set_title(f"With Diffusion (Plane {plane_id})")
        plt.colorbar(im1, ax=axes[1])
        
        # Difference
        diff = proj_with_diff - proj_no_diff
        im2 = axes[2].imshow(diff, aspect='auto', cmap='coolwarm', vmin=-0.5, vmax=0.5)
        axes[2].set_title(f"Difference (Plane {plane_id})")
        plt.colorbar(im2, ax=axes[2])
        
        plt.tight_layout()
        plt.savefig(f"img/diffusion_comparison_plane_{plane_id}.png")
        plt.close(fig)
    
    # Now reconstruct and optimize with both methods
    print("\nReconstructing from projections...")
    # Use the backprojection threshold to get candidate points
    backprojection_threshold = 0.1
    
    # Reconstruction without diffusion
    print("\nReconstructing without diffusion...")
    candidate_points_no_diff = reconstructor_no_diffusion.reconstruct_from_projections(
        projections_no_diffusion, 
        threshold=backprojection_threshold,
        fast_merge=True, 
        snap_to_grid=True
    )
    print(f"Generated {candidate_points_no_diff.shape[0]} candidate points (no diffusion)")
    
    # Reconstruction with diffusion
    print("\nReconstructing with diffusion...")
    candidate_points_with_diff = reconstructor_with_diffusion.reconstruct_from_projections(
        projections_with_diffusion, 
        threshold=backprojection_threshold,
        fast_merge=True, 
        snap_to_grid=True
    )
    print(f"Generated {candidate_points_with_diff.shape[0]} candidate points (with diffusion)")
    
    # Visualize candidate points
    # No diffusion
    candidate_volume_no_diff = torch.zeros(volume_shape, device=device)
    candidate_volume_no_diff[candidate_points_no_diff[:, 0].long(), 
                           candidate_points_no_diff[:, 1].long(), 
                           candidate_points_no_diff[:, 2].long()] = 1.0
    
    fig = visualize_volume(candidate_volume_no_diff.cpu().numpy(), threshold=0.0)
    fig.suptitle("Candidate Points (No Diffusion)")
    fig.savefig("img/diffusion_candidate_points_no_diffusion.png")
    plt.close(fig)
    
    # With diffusion
    candidate_volume_with_diff = torch.zeros(volume_shape, device=device)
    candidate_volume_with_diff[candidate_points_with_diff[:, 0].long(), 
                             candidate_points_with_diff[:, 1].long(), 
                             candidate_points_with_diff[:, 2].long()] = 1.0
    
    fig = visualize_volume(candidate_volume_with_diff.cpu().numpy(), threshold=0.0)
    fig.suptitle("Candidate Points (With Diffusion)")
    fig.savefig("img/diffusion_candidate_points_with_diffusion.png")
    plt.close(fig)
    
    print("\nComparison completed. Check the 'img' directory for output images.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Demonstrate electron diffusion effects in LArTPC reconstruction')
    parser.add_argument('--shape', type=int, default=100, help='Size of the cubic volume (shape x shape x shape)')
    parser.add_argument('--sigma-t', type=float, default=2.0, help='Transverse diffusion coefficient')
    parser.add_argument('--sigma-l', type=float, default=1.0, help='Longitudinal diffusion coefficient')
    parser.add_argument('--attenuation', type=float, default=0.01, help='Electron attenuation coefficient')
    
    args = parser.parse_args()
    
    volume_shape = (args.shape, args.shape, args.shape)
    
    compare_diffusion_effects(
        volume_shape=volume_shape,
        diffusion_sigma_t=args.sigma_t,
        diffusion_sigma_l=args.sigma_l,
        attenuation_coeff=args.attenuation
    ) 