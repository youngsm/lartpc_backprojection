import torch
import numpy as np
import matplotlib.pyplot as plt
import time
import os
import sys

# Add the parent directory to the path to import the package
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from lartpc_reconstruction import LArTPCReconstructor
from lartpc_reconstruction.visualization import visualize_projections

def create_test_volumes():
    """
    Create a set of test volumes with different patterns to test projection consistency.
    """
    # Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Volume shape
    volume_shape = (100, 100, 100)
    volumes = []
    
    # Volume 1: A sphere in the center
    vol1 = torch.zeros(volume_shape, device=device)
    center = torch.tensor([50, 50, 50], device=device)
    radius = 30
    
    # Create coordinate grids
    x = torch.arange(volume_shape[0], device=device)
    y = torch.arange(volume_shape[1], device=device)
    z = torch.arange(volume_shape[2], device=device)
    
    # Create meshgrid
    x_grid, y_grid, z_grid = torch.meshgrid(x, y, z, indexing='ij')
    
    # Calculate distance from center
    distance = torch.sqrt(
        (x_grid - center[0])**2 + 
        (y_grid - center[1])**2 + 
        (z_grid - center[2])**2
    )
    
    # Add sphere to the volume
    vol1[distance <= radius] = 1.0
    volumes.append(vol1)
    
    # Volume 2: A cube in the corner
    vol2 = torch.zeros(volume_shape, device=device)
    vol2[:25, :25, :25] = 1.0
    volumes.append(vol2)
    
    # Volume 3: A diagonal line
    vol3 = torch.zeros(volume_shape, device=device)
    for i in range(min(volume_shape)):
        vol3[i, i, i] = 1.0
    volumes.append(vol3)
    
    return volumes

def test_standardized_projections():
    """
    Test that projections now have standardized sizes regardless of the volume content.
    """
    print("Testing standardized projection sizes...")
    
    # Use CUDA if available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Create test volumes
    test_volumes = create_test_volumes()
    volume_shape = test_volumes[0].shape
    
    # Create reconstructor with three planes
    plane_angles = {0: 0.0, 1: np.pi/3, 2: 2*np.pi/3}
    reconstructor = LArTPCReconstructor(
        volume_shape=volume_shape, 
        device=device,
        plane_angles=plane_angles,
        debug=True
    )
    
    # Project each volume and verify projection sizes
    for i, volume in enumerate(test_volumes):
        print(f"\nProjecting volume {i+1}...")
        projections = reconstructor.project_volume(volume)
        
        # Check if all projections have the correct standardized sizes
        for plane_id, proj in projections.items():
            expected_size = reconstructor.solver.projection_sizes[plane_id]
            actual_size = proj.shape
            
            print(f"  Plane {plane_id}: Expected size {expected_size}, Actual size {actual_size}")
            
            assert expected_size == actual_size, f"Projection size mismatch for plane {plane_id}"
        
        # Visualize the projections
        plt.figure(figsize=(15, 5))
        plt.suptitle(f"Volume {i+1} Projections")
        
        for j, (plane_id, proj) in enumerate(projections.items()):
            plt.subplot(1, 3, j+1)
            plt.imshow(proj.cpu().numpy(), aspect='auto')
            plt.title(f"Plane {plane_id} - Shape: {proj.shape}")
            plt.colorbar()
        
        plt.tight_layout()
        plt.savefig(f"volume_{i+1}_projections.png")
        plt.close()
    
    print("\nAll projection sizes match the expected standardized sizes!")
    
    # Also test with differentiable projections
    print("\nTesting differentiable projections...")
    volume = test_volumes[0]  # Use the first volume
    
    diff_projections = reconstructor.project_volume_differentiable(volume)
    for plane_id, proj in diff_projections.items():
        expected_size = reconstructor.solver.projection_sizes[plane_id]
        actual_size = proj.shape
        
        print(f"  Plane {plane_id}: Expected size {expected_size}, Actual size {actual_size}")
        
        assert expected_size == actual_size, f"Differentiable projection size mismatch for plane {plane_id}"
    
    print("\nAll differentiable projection sizes match the expected standardized sizes!")
    
    # Also test with sparse projection
    print("\nTesting sparse projections...")
    
    # Convert dense volume to sparse
    non_zero_indices = torch.nonzero(volume > 0.5, as_tuple=False)
    non_zero_values = volume[non_zero_indices[:, 0], non_zero_indices[:, 1], non_zero_indices[:, 2]]
    sparse_volume = (non_zero_indices, non_zero_values, volume_shape)
    
    sparse_projections = reconstructor.project_sparse_volume(sparse_volume)
    for plane_id, proj in sparse_projections.items():
        expected_size = reconstructor.solver.projection_sizes[plane_id]
        actual_size = proj.shape
        
        print(f"  Plane {plane_id}: Expected size {expected_size}, Actual size {actual_size}")
        
        assert expected_size == actual_size, f"Sparse projection size mismatch for plane {plane_id}"
    
    print("\nAll sparse projection sizes match the expected standardized sizes!")
    
    # Compare projections from different representations
    plt.figure(figsize=(15, 15))
    plt.suptitle("Comparison of Different Projection Methods")
    
    for plane_id in projections:
        # Dense projection
        plt.subplot(3, 3, plane_id*3 + 1)
        plt.imshow(projections[plane_id].cpu().numpy(), aspect='auto')
        plt.title(f"Plane {plane_id}: Dense")
        plt.colorbar()
        
        # Differentiable projection
        plt.subplot(3, 3, plane_id*3 + 2)
        plt.imshow(diff_projections[plane_id].cpu().numpy(), aspect='auto')
        plt.title(f"Plane {plane_id}: Differentiable")
        plt.colorbar()
        
        # Sparse projection
        plt.subplot(3, 3, plane_id*3 + 3)
        plt.imshow(sparse_projections[plane_id].cpu().numpy(), aspect='auto')
        plt.title(f"Plane {plane_id}: Sparse")
        plt.colorbar()
    
    plt.tight_layout()
    plt.savefig("projection_comparison.png")
    plt.close()
    
    print("\nTest completed successfully! All projection methods now produce standardized sizes.")
    print("Visualization saved to:")
    print("  - volume_1_projections.png, volume_2_projections.png, volume_3_projections.png")
    print("  - projection_comparison.png")

if __name__ == "__main__":
    test_standardized_projections() 