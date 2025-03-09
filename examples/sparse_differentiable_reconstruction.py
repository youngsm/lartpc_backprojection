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

def create_sample_sparse_volume(shape=(100, 100, 100), num_points=1000, device='cuda'):
    """Create a sample sparse 3D volume with random points for testing."""
    # Random coordinates within the volume
    coords = torch.randint(0, min(shape), (num_points, 3), device=device)
    
    # Random values for the points (will be optimized)
    values = torch.rand(num_points, device=device) * 0.5 + 0.5
    
    return coords, values, shape

def run_sparse_differentiable_reconstruction_example():
    """
    Demonstrate differentiable reconstruction with sparse point optimization.
    Only optimizes alpha values for a fixed set of 3D points.
    """
    print("Running sparse differentiable reconstruction example...")
    
    # Use CUDA if available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Create a ground truth sparse volume
    print("Creating ground truth sparse volume...")
    volume_shape = (100, 100, 100)
    
    # For ground truth, create structured points (e.g., a sphere)
    radius = 30
    center = torch.tensor([50, 50, 50], device=device)
    
    # Create points on a sphere surface
    theta = torch.arange(0, np.pi, 0.2, device=device)
    phi = torch.arange(0, 2*np.pi, 0.2, device=device)
    
    theta_grid, phi_grid = torch.meshgrid(theta, phi, indexing='ij')
    
    x = center[0] + radius * torch.sin(theta_grid) * torch.cos(phi_grid)
    y = center[1] + radius * torch.sin(theta_grid) * torch.sin(phi_grid)
    z = center[2] + radius * torch.cos(theta_grid)
    
    # Stack coordinates
    coords = torch.stack([x.flatten(), y.flatten(), z.flatten()], dim=1).long()
    
    # Remove any out-of-bounds points
    mask = ((coords[:, 0] >= 0) & (coords[:, 0] < volume_shape[0]) &
            (coords[:, 1] >= 0) & (coords[:, 1] < volume_shape[1]) &
            (coords[:, 2] >= 0) & (coords[:, 2] < volume_shape[2]))
    
    coords = coords[mask]
    
    # Assign values to the points
    values = torch.ones(coords.shape[0], device=device)
    
    # Create the ground truth sparse volume
    ground_truth_sparse = (coords, values, volume_shape)
    
    print(f"Created ground truth sparse volume with {coords.shape[0]} points")
    
    # Create reconstructor with three planes
    print("Creating reconstructor with 3 planes...")
    plane_angles = {0: 0.0, 1: np.pi/3, 2: 2*np.pi/3}
    reconstructor = LArTPCReconstructor(
        volume_shape=volume_shape, 
        device=device,
        plane_angles=plane_angles,
        debug=True
    )
    
    # Create projections from ground truth
    print("Creating projections from ground truth...")
    projections = reconstructor.project_sparse_volume_differentiable(ground_truth_sparse)
    
    # Visualize ground truth volume
    print("Visualizing ground truth sparse volume...")
    # Convert sparse to dense for visualization
    dense_ground_truth = torch.zeros(volume_shape, device=device)
    dense_ground_truth[coords[:, 0], coords[:, 1], coords[:, 2]] = values
    visualize_volume(dense_ground_truth.cpu().numpy())
    
    # Visualize projections
    visualize_projections(
        {k: v.cpu().detach().numpy() for k, v in projections.items()}
    )
    
    # Create candidate points from backprojected lines
    # In a real scenario, these would come from actual backprojection
    # For this example, we'll:
    # 1. Create a larger set of points that includes the ground truth
    # 2. Plus some additional noise points
    
    print("Creating candidate points from backprojection...")
    
    # Create more candidate points (including some noise)
    num_candidate_points = coords.shape[0] * 4  # 4x the ground truth points
    
    # Sample additional random points
    additional_coords = torch.randint(0, min(volume_shape), (num_candidate_points - coords.shape[0], 3), device=device)
    
    # Combine with ground truth coordinates
    candidate_coords = torch.cat([coords, additional_coords], dim=0)
    
    print(f"Created {candidate_coords.shape[0]} candidate points")
    
    # Use the new optimize_sparse_point_intensities method
    print("Starting optimization of point intensities...")
    
    # Optimization parameters
    num_iterations = 300
    learning_rate = 0.01
    pruning_threshold = 0.05
    pruning_interval = 200
    l1_weight = 0.1
    
    # Run optimization
    optimized_coords, optimized_values, loss_history, num_points_history = reconstructor.optimize_sparse_point_intensities(
        candidate_points=candidate_coords,
        target_projections=projections,
        num_iterations=num_iterations,
        lr=learning_rate,
        pruning_threshold=pruning_threshold,
        pruning_interval=pruning_interval,
        l1_weight=l1_weight
    )
    
    # Visualize final reconstructed volume
    print("Optimization complete, visualizing results...")
    
    # Convert final sparse to dense for visualization
    final_dense = torch.zeros(volume_shape, device=device)
    final_dense[optimized_coords[:, 0], optimized_coords[:, 1], optimized_coords[:, 2]] = optimized_values
    
    visualize_volume(final_dense.cpu().numpy(), threshold=0.1)
    
    # Calculate accuracy metrics
    # Find overlap between ground truth and optimized points
    gt_points_set = {(coords[i, 0].item(), coords[i, 1].item(), coords[i, 2].item()) for i in range(coords.shape[0])}
    optimized_points_set = {(optimized_coords[i, 0].item(), optimized_coords[i, 1].item(), optimized_coords[i, 2].item()) 
                          for i in range(optimized_coords.shape[0]) if optimized_values[i].item() > 0.1}
    
    common_points = gt_points_set.intersection(optimized_points_set)
    precision = len(common_points) / max(1, len(optimized_points_set))
    recall = len(common_points) / max(1, len(gt_points_set))
    f1_score = 2 * precision * recall / max(1e-8, precision + recall)
    
    print(f"Point-based metrics:")
    print(f"  Ground truth points: {len(gt_points_set)}")
    print(f"  Optimized points: {len(optimized_points_set)}")
    print(f"  Common points: {len(common_points)}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1 Score: {f1_score:.4f}")
    
    # Visualize comparison of projections
    final_projections = reconstructor.project_sparse_volume_differentiable(
        (optimized_coords, optimized_values, volume_shape)
    )
    
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
    plt.savefig("sparse_projection_comparison.png")
    plt.close()
    
    # Plot loss curve
    plt.figure(figsize=(10, 6))
    plt.plot(loss_history)
    plt.title('Loss During Optimization')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.yscale('log')
    plt.grid(True)
    plt.savefig("sparse_optimization_loss.png")
    plt.close()
    
    # Plot number of points over iterations
    pruning_iterations = [i * pruning_interval for i in range(len(num_points_history))]
    plt.figure(figsize=(10, 6))
    plt.plot(pruning_iterations, num_points_history, 'o-')
    plt.title('Number of Points During Optimization')
    plt.xlabel('Iteration')
    plt.ylabel('Number of Points')
    plt.grid(True)
    plt.savefig("sparse_point_count.png")
    plt.close()
    
    print("Example complete! Results saved as sparse_projection_comparison.png, "
          "sparse_optimization_loss.png, and sparse_point_count.png")

if __name__ == "__main__":
    run_sparse_differentiable_reconstruction_example() 