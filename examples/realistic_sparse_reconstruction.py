from matplotlib.colors import LogNorm
import torch
import numpy as np
import matplotlib.pyplot as plt
import time
import os
import sys
import math 

# Add the parent directory to the path to import the package
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from lartpc_reconstruction import LArTPCReconstructor
from lartpc_reconstruction.visualization import (
    visualize_volume, 
    visualize_projections, 
    visualize_original_vs_reconstructed
)
tracks = [
    ((20, 20, 20), (100, 100, 100)),  # Diagonal line
    ((20, 50, 50), (100, 50, 50)),  # Horizontal line
]


# Example: Create a helix
def create_helix(start=(40, 40, 20), radius=20, height=80, turns=3, num_points=100):
    t = np.linspace(0, turns * 2 * np.pi, num_points)
    x = start[0] + radius * np.cos(t)
    y = start[1] + radius * np.sin(t)
    z = start[2] + height * t / (turns * 2 * np.pi)
    return np.column_stack((x, y, z))


# Example: Create a sine wave in 3D space
def create_sine_wave(
    start=(20, 60, 60), length=80, amplitude=20, periods=2, num_points=100
):
    t = np.linspace(0, 1, num_points)
    x = start[0] + length * t
    y = start[1] + amplitude * np.sin(periods * 2 * np.pi * t)
    z = start[2] + amplitude * np.cos(periods * 2 * np.pi * t)
    return np.column_stack((x, y, z))


# Create a helix
helix_points = create_helix(num_points=500)

# Create a sine wave
sine_points = create_sine_wave(num_points=500)

track_points = []
for start, end in tracks:
    # Create multiple points along the line to ensure continuity
    start_point = np.array(start)
    end_point = np.array(end)
    line_vec = end_point - start_point
    line_length = np.linalg.norm(line_vec)
    num_points = int(np.ceil(line_length)) + 1  # One point per mm
    t_values = np.linspace(0, 1, num_points)
    line_points = start_point + np.outer(t_values, line_vec)
    track_points.append(line_points)
track_points = np.vstack(track_points)
input_points = track_points

# Option 3: Use parametric curves (helix or sine wave)
input_points = helix_points
input_points = sine_points

# Option 4: Combine multiple sets
input_points = np.vstack([helix_points, sine_points, track_points])

def create_volume_from_points(points, volume_shape, radius=0.0, device="cuda"):
    """
    Create a 3D volume from a set of points.
    Each point is expanded to a sphere with the given radius.

    Args:
        points (numpy.ndarray): Array of shape (N, 3) or (N, 4) containing point coordinates
                               If shape is (N, 4), the 4th column is used as point values
        volume_shape (tuple): Shape of the volume (x, y, z)
        radius (float): Radius of each point in voxels
        device (str): Device to use ('cuda' or 'cpu')

    Returns:
        tuple: (sparse_volume, dense_volume, original_points)
               - sparse_volume is a tuple of (coords, values, shape)
               - dense_volume is a torch.Tensor
               - original_points is the input points in the correct format
    """
    # Check if points include values
    has_values = points.shape[1] > 3

    # Extract coordinates and values
    if has_values:
        point_coords = points[:, :3]
        point_values = torch.tensor(points[:, 3], dtype=torch.float32, device=device)
    else:
        point_coords = points
        point_values = torch.ones(len(points), dtype=torch.float32, device=device)

    # Convert points to tensor
    points_tensor = torch.tensor(point_coords, dtype=torch.float32, device=device)

    # Ensure points are within volume bounds
    points_tensor = torch.clamp(
        points_tensor,
        min=torch.zeros(3, device=device),
        max=torch.tensor(volume_shape, device=device) - 1,
    )

    # Round to get voxel coordinates
    voxel_coords = torch.round(points_tensor).long()

    # If radius > 0, expand each point to a sphere
    if radius > 0:
        # Create offsets for points in a sphere
        r = int(math.ceil(radius))
        offsets = []
        for dx in range(-r, r + 1):
            for dy in range(-r, r + 1):
                for dz in range(-r, r + 1):
                    dist = math.sqrt(dx**2 + dy**2 + dz**2)
                    if dist <= radius:
                        offsets.append((dx, dy, dz, dist))

        # Convert offsets to tensor
        offset_coords = torch.tensor(
            [(o[0], o[1], o[2]) for o in offsets], device=device
        )

        # For each point, add all offsets
        expanded_coords = []
        expanded_values = []

        for i, point in enumerate(voxel_coords):
            # Add offsets to the point
            sphere_coords = point.unsqueeze(0) + offset_coords

            # Calculate values based on distance from center and point value
            distances = torch.tensor([o[3] for o in offsets], device=device)
            values = point_values[i] * torch.exp(-((distances / radius) ** 2) / 2)

            # Ensure coordinates are within bounds
            valid_mask = (
                (sphere_coords[:, 0] >= 0)
                & (sphere_coords[:, 0] < volume_shape[0])
                & (sphere_coords[:, 1] >= 0)
                & (sphere_coords[:, 1] < volume_shape[1])
                & (sphere_coords[:, 2] >= 0)
                & (sphere_coords[:, 2] < volume_shape[2])
            )

            expanded_coords.append(sphere_coords[valid_mask])
            expanded_values.append(values[valid_mask])

        # Combine all expanded coordinates
        all_coords = torch.cat(expanded_coords, dim=0)
        all_values = torch.cat(expanded_values, dim=0)
    else:
        # Just use the original coordinates with their values
        all_coords = voxel_coords
        all_values = point_values

    # Create a unique identifier for each voxel
    voxel_ids = (
        all_coords[:, 0] * volume_shape[1] * volume_shape[2]
        + all_coords[:, 1] * volume_shape[2]
        + all_coords[:, 2]
    )

    # Find unique voxels and take maximum value for each
    unique_ids, inverse_indices = torch.unique(voxel_ids, return_inverse=True)

    # Create output arrays
    unique_coords = torch.zeros((len(unique_ids), 3), dtype=torch.long, device=device)
    unique_values = torch.zeros(len(unique_ids), device=device)

    # For each unique ID, find the maximum value
    for i, id in enumerate(unique_ids):
        mask = voxel_ids == id
        unique_coords[i] = all_coords[mask][0]  # Take the first matching coordinate
        unique_values[i] = torch.sum(all_values[mask])  # Take the maximum value

    # Create the sparse representation
    sparse_volume = (unique_coords, unique_values, volume_shape)

    # Create the dense representation
    dense_volume = torch.zeros(volume_shape, device=device)
    dense_volume[unique_coords[:, 0], unique_coords[:, 1], unique_coords[:, 2]] = (
        unique_values
    )

    return sparse_volume, dense_volume, points

def run_realistic_sparse_reconstruction(optimization_method='adam'):
    """
    Demonstrate a realistic sparse point-based differentiable reconstruction.
    Uses the actual backprojection process to get candidate points.
    
    Args:
        optimization_method (str): The optimization method to use, either 'adam' or 'admm'
    """
    print(f"Running realistic sparse point-based differentiable reconstruction using {optimization_method.upper()}...")
    
    # Use CUDA if available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Create a 3D volume with a simple structure (e.g., a cube)
    print("Creating ground truth volume...")
    volume_shape = (100, 100, 100)
    
    # # Create a cube in the center of the volume
    # cube_size = 30
    # start_idx = (volume_shape[0] - cube_size) // 2
    # end_idx = start_idx + cube_size
    
    # # Create dense ground truth
    # ground_truth = torch.zeros(volume_shape, device=device)
    # ground_truth[start_idx:end_idx, start_idx:end_idx, start_idx:end_idx] = 1.0
    
    # # Add some random noise to make it more interesting
    # noise = torch.randn(volume_shape, device=device) * 0.05
    # ground_truth = torch.clamp(ground_truth + noise, 0, 1)
    
    # # Convert to sparse representation for efficiency
    # non_zero_indices = torch.nonzero(ground_truth > 0.1, as_tuple=False)
    # non_zero_values = ground_truth[non_zero_indices[:, 0], non_zero_indices[:, 1], non_zero_indices[:, 2]]
    
    # Create sparse ground truth
    ground_truth_sparse, ground_truth, original_points = create_volume_from_points(input_points, volume_shape, radius=0.0, device=device)
    print(f"Created ground truth with {original_points.shape[0]} non-zero points")
    
    # Create reconstructor with three planes
    print("Creating reconstructor with 3 planes...")
    plane_angles = {0: 0.0, 1: -np.pi/6, 2: np.pi/6}  # Three planes at different angles
    reconstructor = LArTPCReconstructor(
        volume_shape=volume_shape, 
        device=device,
        plane_angles=plane_angles,
        debug=False
    )
    
    # Generate projections from ground truth
    print("Generating 2D projections from ground truth...")
    projections = reconstructor.project_sparse_volume_differentiable(ground_truth_sparse)
    
    # Visualize ground truth and projections
    print("Visualizing ground truth and projections...")
    visualize_volume(ground_truth.cpu().numpy(), threshold=0.0).savefig(f"img/realistic_ground_truth_{optimization_method}.png")
    visualize_projections({k: v.cpu().detach().numpy() for k, v in projections.items()}).savefig(f"img/realistic_projections_{optimization_method}.png")
    
    # Threshold the projections to simulate detector hits
    print("Thresholding projections to simulate detector hits...")
    thresholded_projections = projections

    # threshold = 0.1
    # thresholded_projections = {}

    # for plane_id, projection in projections.items():
    #     thresholded_projections[plane_id] = (projection > threshold).float()
    
    # Use the standard reconstruction process to get candidate points
    print("Reconstructing candidate points using standard method...")
    backprojection_threshold = 0.1
    candidate_points = reconstructor.reconstruct_from_projections(
        thresholded_projections, 
        threshold=backprojection_threshold,
        fast_merge=True, 
        snap_to_grid=True
    )
    
    print(f"Generated {candidate_points.shape[0]} candidate points from backprojection")
    
    # Visualize candidate points
    candidate_volume = torch.zeros(volume_shape, device=device)
    candidate_volume[candidate_points[:, 0].long(), candidate_points[:, 1].long(), candidate_points[:, 2].long()] = 1.0
    visualize_volume(candidate_volume.cpu().numpy(), threshold=0.0).savefig(f"img/realistic_candidate_points_{optimization_method}.png")
    
    # Optimize the intensities of these candidate points
    print(f"Optimizing intensities of candidate points using {optimization_method.upper()}...")
    
    # Convert candidate points to long for indexing
    candidate_points_long = candidate_points.long()
    
    # Optimization parameters
    num_iterations = 5000 if optimization_method == 'adam' else 200  # ADMM typically needs fewer iterations
    learning_rate = 0.01
    pruning_threshold = 0.01
    pruning_interval = 250 if optimization_method == 'adam' else 10000
    l1_weight = 0.01
    
    # Run optimization with the specified method
    if optimization_method == 'adam':
        # Use Adam (SGD) optimization
        optimized_coords, optimized_values, loss_history, num_points_history = reconstructor.optimize_sparse_point_intensities(
            candidate_points=candidate_points_long,
            target_projections=projections,
            num_iterations=num_iterations,
            lr=learning_rate,
            pruning_threshold=pruning_threshold,
            pruning_interval=pruning_interval,
            l1_weight=l1_weight,
        )
    else:  # 'admm'
        # Use ADMM optimization
        rho = 1.0
        alpha = 1.0
        relaxation = 1.0
        optimized_coords, optimized_values, loss_history, num_points_history = reconstructor.optimize_sparse_point_intensities_admm(
            candidate_points=candidate_points_long,
            target_projections=projections,
            num_iterations=num_iterations,
            rho=rho,
            alpha=alpha,
            pruning_threshold=pruning_threshold,
            pruning_interval=pruning_interval,
            l1_weight=l1_weight,
            relaxation=relaxation
        )
    
    # Convert final result to dense for visualization
    final_volume = torch.zeros(volume_shape, device=device)
    final_volume[optimized_coords[:, 0], optimized_coords[:, 1], optimized_coords[:, 2]] = optimized_values
    
    # Calculate metrics
    metrics = reconstructor.evaluate_reconstruction(ground_truth, final_volume, threshold=0.0)
    
    # Visualize final results
    print("Visualizing final results...")
    visualize_volume(final_volume.cpu().numpy(), threshold=0.0).savefig(f"img/realistic_final_volume_{optimization_method}.png")
    
    # Compare original and reconstructed
    fig = visualize_original_vs_reconstructed(
        ground_truth.cpu().numpy(), 
        final_volume.cpu().numpy(), 
        # threshold=0.1
    )
    fig.savefig(f"img/realistic_original_vs_reconstructed_{optimization_method}.png")
    plt.close(fig)
    
    # Visualize comparison of projections
    final_projections = reconstructor.project_sparse_volume_differentiable(
        (optimized_coords, optimized_values, volume_shape)
    )
    
    # Create a figure to compare original and final projections
    fig, axes = plt.subplots(3, 2, figsize=(12, 12))
    
    for i, plane_id in enumerate(projections):
        # Original projection
        orig_proj = projections[plane_id].cpu().detach().numpy()
        im = axes[i, 0].imshow(orig_proj, aspect='auto', cmap='viridis', interpolation='none', norm=LogNorm())
        axes[i, 0].set_title(f"Original Projection (Plane {plane_id})")
        plt.colorbar(im, ax=axes[i, 0])
        
        # Optimized projection
        final_proj = final_projections[plane_id].cpu().detach().numpy()
        im = axes[i, 1].imshow(final_proj, aspect='auto', cmap='viridis', interpolation='none', norm=LogNorm())
        axes[i, 1].set_title(f"Optimized Projection (Plane {plane_id})")
        plt.colorbar(im, ax=axes[i, 1])
    
    plt.tight_layout()
    plt.savefig(f"img/realistic_projection_comparison_{optimization_method}.png")
    plt.close()
    
    # Calculate how many original points are included in the reconstruction
    # An original point is considered "included" if there's a reconstructed point close to it
    inclusion_threshold = 1.1  # Maximum distance for a point to be considered included
    original_points_array = torch.stack(original_points) if isinstance(original_points, list) else original_points
    total_original_points = len(original_points_array)
    
    included_count = 0
    if len(optimized_coords) > 0:
        for orig_point in original_points_array:
            # Calculate distances to all reconstructed points
            distances = torch.norm(optimized_coords.float().cpu() - orig_point[None, ...], dim=1)
            # If any reconstructed point is close enough, consider the original point included
            if torch.any(distances < inclusion_threshold):
                included_count += 1
    
    inclusion_percentage = 100 * included_count / total_original_points if total_original_points > 0 else 0
    
    print(f"\nOriginal points statistics:")
    print(f"  Total original points: {total_original_points}")
    print(f"  Original points included in reconstruction: {included_count} ({inclusion_percentage:.2f}%)")
    
    # Add this information to the title of the loss curve
    # Plot loss curve
    plt.figure(figsize=(10, 6))
    plt.plot(loss_history)
    plt.title(f'Loss During {optimization_method.upper()} Optimization\n'
              f'Original: {total_original_points} points, '
              f'Included: {included_count} ({inclusion_percentage:.2f}%)')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.yscale('log')
    plt.grid(True)
    plt.savefig(f"img/realistic_optimization_loss_{optimization_method}.png")
    plt.close()
    
    # Plot number of points over iterations
    pruning_iterations = [i * pruning_interval for i in range(len(num_points_history))]
    plt.figure(figsize=(10, 6))
    plt.plot(pruning_iterations, num_points_history, 'o-')
    plt.title(f'Number of Points During {optimization_method.upper()} Optimization\n'
              f'Original: {total_original_points} points, '
              f'Included: {included_count} ({inclusion_percentage:.2f}%)')
    plt.xlabel('Iteration')
    plt.ylabel('Number of Points')
    plt.grid(True)
    plt.savefig(f"img/realistic_points_history_{optimization_method}.png")
    plt.close()
    
    # Print a summary of the metrics, including non-zero PSNR
    print(f"\nMetrics for {optimization_method.upper()} reconstruction:")
    # Standard metrics
    standard_metrics = ['iou', 'dice', 'precision', 'recall', 'f1', 'mse', 'psnr']
    for metric_name in standard_metrics:
        if metric_name in metrics:
            print(f"  {metric_name}: {metrics[metric_name]:.6f}")
    
    # Non-zero metrics with emphasis
    special_metrics = ['mse_nonzero', 'psnr_nonzero']
    if any(m in metrics for m in special_metrics):
        print("\n  Non-zero region metrics (only considering pixels where target > 0):")
        for metric_name in special_metrics:
            if metric_name in metrics:
                print(f"    {metric_name}: {metrics[metric_name]:.6f}")
        
    return metrics, final_volume

# If this script is run directly, run the example
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Run realistic sparse reconstruction example')
    parser.add_argument('--method', type=str, choices=['adam', 'admm', 'both'], default='both',
                        help='Optimization method to use: adam, admm, or both')
    args = parser.parse_args()
    
    # Create img directory if it doesn't exist
    os.makedirs('img', exist_ok=True)
    
    # Define inputs for visualization
    input_points = []
    
    # Create example structures
    helix_points = create_helix(start=(40, 40, 20), radius=20, height=60, turns=3, num_points=500)
    sine_points = create_sine_wave(start=(20, 60, 60), length=80, amplitude=20, periods=2, num_points=500)
    second_sine = create_sine_wave(start=(20, 30, 30), length=80, amplitude=15, periods=3, num_points=500)
    
    # Add to input points
    input_points.extend(helix_points)
    input_points.extend(sine_points)
    input_points.extend(second_sine)
    
    # Convert to tensor
    input_points = torch.tensor(input_points, dtype=torch.float32)
    
    if args.method == 'adam':
        # Run with Adam only
        run_realistic_sparse_reconstruction(optimization_method='adam')
    elif args.method == 'admm':
        # Run with ADMM only
        run_realistic_sparse_reconstruction(optimization_method='admm')
    else:  # 'both'
        # Run with both methods and compare
        print("\n========== Running with Adam optimization ==========\n")
        adam_metrics, adam_volume = run_realistic_sparse_reconstruction(optimization_method='adam')
        
        print("\n========== Running with ADMM optimization ==========\n")
        admm_metrics, admm_volume = run_realistic_sparse_reconstruction(optimization_method='admm')
        
        # Compare metrics
        print("\n========== Comparison of Metrics ==========")
        print(f"{'Metric':<20} {'Adam':<15} {'ADMM':<15} {'Difference':<15}")
        print("-" * 65)
        
        # First standard metrics
        standard_metrics = ['iou', 'dice', 'precision', 'recall', 'f1', 'mse', 'psnr']
        for metric in standard_metrics:
            if metric in adam_metrics:
                adam_val = adam_metrics[metric]
                admm_val = admm_metrics[metric]
                diff = admm_val - adam_val
                print(f"{metric:<20} {adam_val:<15.6f} {admm_val:<15.6f} {diff:<15.6f}")
        
        # Then non-zero metrics with emphasis
        special_metrics = ['mse_nonzero', 'psnr_nonzero']
        if any(m in adam_metrics for m in special_metrics):
            print("\nNon-zero region metrics (only considering pixels where target > 0):")
            print("-" * 65)
            for metric in special_metrics:
                if metric in adam_metrics:
                    adam_val = adam_metrics[metric]
                    admm_val = admm_metrics[metric]
                    diff = admm_val - adam_val
                    print(f"{metric:<20} {adam_val:<15.6f} {admm_val:<15.6f} {diff:<15.6f}")
        
        # Create a direct visual comparison of the two reconstructions and metrics
        print("\nGenerating side-by-side comparison of reconstructions...")
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))
        
        im1 = axes[0].imshow(np.max(adam_volume.cpu().numpy(), axis=0), cmap='viridis')
        axes[0].set_title("Adam Reconstruction")
        plt.colorbar(im1, ax=axes[0])
        
        im2 = axes[1].imshow(np.max(admm_volume.cpu().numpy(), axis=0), cmap='viridis')
        axes[1].set_title("ADMM Reconstruction")
        plt.colorbar(im2, ax=axes[1])
        
        plt.tight_layout()
        plt.savefig("img/adam_vs_admm_comparison.png")
        plt.close()
        
        # Add a special visualization for PSNR comparison
        if ('psnr' in adam_metrics and 'psnr' in admm_metrics and 
            'psnr_nonzero' in adam_metrics and 'psnr_nonzero' in admm_metrics):
            print("\nGenerating PSNR comparison chart...")
            fig, ax = plt.subplots(figsize=(12, 6))
            
            # Data to plot
            methods = ['Adam', 'ADMM']
            psnr_standard = [adam_metrics['psnr'], admm_metrics['psnr']]
            psnr_nonzero = [adam_metrics['psnr_nonzero'], admm_metrics['psnr_nonzero']]
            
            # Set up width and positions
            x = np.arange(len(methods))
            width = 0.35
            
            # Create bars
            rects1 = ax.bar(x - width/2, psnr_standard, width, label='Standard PSNR', color='skyblue')
            rects2 = ax.bar(x + width/2, psnr_nonzero, width, label='Non-zero PSNR', color='orange')
            
            # Add labels, title, and legend
            ax.set_ylabel('PSNR (dB)')
            ax.set_title('PSNR Comparison: All Pixels vs. Non-Zero Pixels Only')
            ax.set_xticks(x)
            ax.set_xticklabels(methods)
            ax.legend()
            
            # Add value labels on bars
            def autolabel(rects):
                for rect in rects:
                    height = rect.get_height()
                    ax.annotate(f'{height:.2f}',
                                xy=(rect.get_x() + rect.get_width()/2, height),
                                xytext=(0, 3),
                                textcoords="offset points",
                                ha='center', va='bottom')
            
            autolabel(rects1)
            autolabel(rects2)
            
            plt.tight_layout()
            plt.savefig("img/adam_vs_admm_psnr_comparison.png")
            plt.close()
        
        print("\nComparison completed. Check the 'img' directory for output images.") 