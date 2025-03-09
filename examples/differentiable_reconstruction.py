import torch
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import time
import os
import sys

# Add the parent directory to the path to import the package
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from lartpc_reconstruction import LArTPCReconstructor
from lartpc_reconstruction.visualization import visualize_volume, visualize_projections

def create_sample_volume(shape=(100, 100, 100), num_spheres=3, device='cuda'):
    """Create a sample 3D volume with spheres for testing."""
    volume = torch.zeros(shape, device=device)
    
    # Create a few spheres in the volume
    for _ in range(num_spheres):
        # Random center position
        center = torch.randint(20, 80, (3,), device=device)
        # Random radius
        radius = torch.randint(5, 15, (1,), device=device).item()
        
        # Create coordinate grids
        x = torch.arange(shape[0], device=device)
        y = torch.arange(shape[1], device=device)
        z = torch.arange(shape[2], device=device)
        
        # Create meshgrid
        x_grid, y_grid, z_grid = torch.meshgrid(x, y, z, indexing='ij')
        
        # Calculate distance from center
        distance = torch.sqrt(
            (x_grid - center[0])**2 + 
            (y_grid - center[1])**2 + 
            (z_grid - center[2])**2
        )
        
        # Add sphere with smooth edges to the volume
        volume += torch.exp(-(distance**2) / (2 * (radius/2)**2))
    
    # Normalize to 0-1 range
    volume = volume / volume.max()
    
    return volume

def run_differentiable_reconstruction_example():
    """Demonstrate differentiable reconstruction for optimization."""
    print("Running differentiable reconstruction example...")
    
    # Use CUDA if available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Create a sample volume (ground truth)
    print("Creating sample volume...")
    volume_shape = (100, 100, 100)
    ground_truth = create_sample_volume(volume_shape, num_spheres=3, device=device)
    ground_truth.requires_grad_(False)  # No need for gradients on ground truth
    
    # Create reconstructor with three planes
    print("Creating reconstructor with 3 planes...")
    plane_angles = {0: 0.0, 1: np.pi/3, 2: 2*np.pi/3}
    reconstructor = LArTPCReconstructor(
        volume_shape=volume_shape, 
        device=device,
        plane_angles=plane_angles,
        debug=True
    )
    
    # Create projections from ground truth using differentiable projection
    print("Creating projections from ground truth...")
    projections = reconstructor.project_volume_differentiable(ground_truth)
    
    # Visualize ground truth volume and projections
    print("Visualizing ground truth volume and projections...")
    visualize_volume(ground_truth.cpu().detach().numpy())
    visualize_projections(
        {k: v.cpu().detach().numpy() for k, v in projections.items()}
    )
    
    # Create a random initial volume to optimize
    print("Creating initial random volume for optimization...")
    initial_volume = torch.rand(volume_shape, device=device) * 0.1
    initial_volume.requires_grad_(True)  # We need gradients for optimization
    
    # Visualize initial volume
    visualize_volume(initial_volume.cpu().detach().numpy(), threshold=0.01)
    
    # Create optimizer
    optimizer = optim.Adam([initial_volume], lr=0.01)
    
    # Optimize the volume to match the projections
    print("Starting optimization...")
    num_iterations = 50
    loss_values = []
    
    for iteration in range(num_iterations):
        start_time = time.time()
        
        # Forward pass: project current volume
        current_projections = reconstructor.project_volume_differentiable(initial_volume)
        
        # Calculate loss (MSE between original and current projections)
        loss = 0
        for plane_id in projections:
            loss += torch.mean((projections[plane_id] - current_projections[plane_id])**2)
        
        # Add L1 regularization for sparsity
        l1_reg = 0.01 * torch.mean(torch.abs(initial_volume))
        loss += l1_reg
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Apply constraints: clamp values to [0, 1] range
        with torch.no_grad():
            initial_volume.clamp_(0, 1)
        
        end_time = time.time()
        loss_values.append(loss.item())
        
        # Print progress
        if (iteration + 1) % 5 == 0 or iteration == 0:
            print(f"Iteration {iteration+1}/{num_iterations}, Loss: {loss.item():.6f}, Time: {end_time - start_time:.2f}s")
    
    # Visualize final reconstructed volume
    print("Optimization complete, visualizing results...")
    final_volume = initial_volume.detach()
    visualize_volume(final_volume.cpu().numpy(), threshold=0.1)
    
    # Visualize comparison of projections
    final_projections = reconstructor.project_volume_differentiable(final_volume)
    
    # Create a figure to compare original and final projections
    fig, axes = plt.subplots(3, 2, figsize=(12, 12))
    
    for i, plane_id in enumerate(projections):
        # Original projection
        orig_proj = projections[plane_id].cpu().detach().numpy()
        im = axes[i, 0].imshow(orig_proj, aspect='auto')
        axes[i, 0].set_title(f"Original Projection (Plane {plane_id})")
        plt.colorbar(im, ax=axes[i, 0])
        
        # Optimized projection
        final_proj = final_projections[plane_id].cpu().detach().numpy()
        im = axes[i, 1].imshow(final_proj, aspect='auto')
        axes[i, 1].set_title(f"Optimized Projection (Plane {plane_id})")
        plt.colorbar(im, ax=axes[i, 1])
    
    plt.tight_layout()
    plt.savefig("projection_comparison.png")
    plt.close()
    
    # Plot loss curve
    plt.figure(figsize=(10, 6))
    plt.plot(loss_values)
    plt.title('Loss During Optimization')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.yscale('log')
    plt.grid(True)
    plt.savefig("optimization_loss.png")
    plt.close()
    
    print("Example complete! Results saved as projection_comparison.png and optimization_loss.png")

if __name__ == "__main__":
    run_differentiable_reconstruction_example() 