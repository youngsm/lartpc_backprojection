import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from time import time
import math

# Add the parent directory to the path to import the package
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from lartpc_reconstruction import (
    LArTPCReconstructor,
    Line3D,
    visualize_volume,
    visualize_projections,
    visualize_lines_and_intersections,
    visualize_original_vs_reconstructed
)

def create_line_volume(start_point=(20, 20, 20), end_point=(100, 100, 100), volume_shape=(128, 128, 128), device='cuda'):
    """
    Create a 3D volume with a single straight line from start_point to end_point.
    Returns both sparse and dense representations.
    
    Args:
        start_point (tuple): Starting point of the line (x, y, z)
        end_point (tuple): Ending point of the line (x, y, z)
        volume_shape (tuple): Shape of the volume (x, y, z)
        device (str): Device to use ('cuda' or 'cpu')
        
    Returns:
        tuple: (sparse_volume, dense_volume, original_points)
               - sparse_volume is a tuple of (coords, values, shape)
               - dense_volume is a torch.Tensor
               - original_points is a numpy array of points along the line
    """
    # Convert start and end points to tensors
    start = torch.tensor(start_point, dtype=torch.float32, device=device)
    end = torch.tensor(end_point, dtype=torch.float32, device=device)
    
    # Calculate line parameters
    line_vec = end - start
    line_length = torch.norm(line_vec)
    line_dir = line_vec / line_length
    
    # Determine number of points needed for a continuous line
    # We want to ensure there's at least one point per voxel along the line
    num_points = int(math.ceil(line_length.item())) + 1
    
    # Generate points along the line
    t_values = torch.linspace(0, 1, num_points, device=device)
    line_points = start.unsqueeze(0) + line_dir.unsqueeze(0) * t_values.unsqueeze(1) * line_length
    
    # Round to nearest voxel coordinates
    voxel_coords = torch.round(line_points).long()
    
    # Ensure all coordinates are within bounds
    voxel_coords = torch.clamp(
        voxel_coords,
        min=torch.zeros(3, dtype=torch.long, device=device),
        max=torch.tensor(volume_shape, dtype=torch.long, device=device) - 1
    )
    
    # Create a unique identifier for each point to remove duplicates
    voxel_ids = (
        voxel_coords[:, 0] * volume_shape[1] * volume_shape[2] +
        voxel_coords[:, 1] * volume_shape[2] +
        voxel_coords[:, 2]
    )
    
    # Get unique voxels
    unique_ids, inverse_indices = torch.unique(voxel_ids, return_inverse=True)
    
    # Create sparse volume representation
    unique_coords = torch.zeros((len(unique_ids), 3), dtype=torch.long, device=device)
    unique_values = torch.ones(len(unique_ids), device=device)  # All voxels have value 1.0
    
    # Get coordinates for each unique voxel
    for i, voxel_id in enumerate(unique_ids):
        mask = (voxel_ids == voxel_id)
        unique_coords[i] = voxel_coords[mask][0]  # Take first occurrence
    
    # Create dense volume
    dense_volume = torch.zeros(volume_shape, device=device)
    dense_volume[unique_coords[:, 0], unique_coords[:, 1], unique_coords[:, 2]] = unique_values
    
    # Create sparse volume representation
    sparse_volume = (unique_coords, unique_values, volume_shape)
    
    # Return original line points as numpy array for visualization
    original_points = line_points.cpu().numpy()
    
    print(f"Created line with {len(unique_ids)} voxels from {start_point} to {end_point}")
    return sparse_volume, dense_volume, original_points

def run_line_example(debug=True, fast_merge=True):
    """
    Create a single line and run the full reconstruction pipeline on it.
    
    Args:
        debug (bool): Whether to print detailed debug information
        fast_merge (bool): Whether to use fast merging mode for intersection clustering
    """
    print("=== Running line example ===")
    
    # Check if CUDA is available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Set parameters
    volume_shape = (128, 128, 128)
    start_point = (20, 20, 20)
    end_point = (100, 100, 100)
    tolerance = 1.5
    merge_tolerance = 2.0
    
    # 1. Create a line volume
    print("\n1. Creating line volume...")
    start_time = time()
    
    sparse_volume, dense_volume, original_points = create_line_volume(
        start_point=start_point,
        end_point=end_point,
        volume_shape=volume_shape,
        device=device
    )
    
    print(f"Created volume in {time() - start_time:.2f} seconds")
    
    # Visualize the volume
    fig = visualize_volume(dense_volume)
    fig.savefig('line_volume.png')
    plt.close(fig)
    
    # 2. Create the reconstructor
    print("\n2. Creating reconstructor...")
    reconstructor = LArTPCReconstructor(
        volume_shape=volume_shape,
        tolerance=tolerance,
        merge_tolerance=merge_tolerance,
        device=device,
        debug=debug
    )
    
    # 3. Project the volume to 2D projections
    print("\n3. Projecting volume to 2D...")
    start_time = time()
    
    # Use sparse representation for projection
    projections = reconstructor.project_volume(sparse_volume)
    
    print(f"Projected volume in {time() - start_time:.2f} seconds")
    
    # Visualize the projections
    fig = visualize_projections(projections)
    fig.savefig('line_projections.png')
    plt.close(fig)
    
    # 4. Reconstruct 3D points from projections
    print("\n4. Reconstructing 3D points from projections...")
    start_time = time()
    reconstructed_points = reconstructor.reconstruct_from_projections(projections, threshold=0.1, fast_merge=fast_merge)
    print(f"Reconstructed {reconstructed_points.size(0)} points in {time() - start_time:.2f} seconds")
    
    # Visualize original vs reconstructed points
    if len(reconstructed_points) > 0:
        fig = visualize_original_vs_reconstructed(
            original_points,
            reconstructed_points.cpu().numpy(),
            volume_shape=volume_shape
        )
        fig.savefig('line_reconstruction.png')
        plt.close(fig)
        
        # Calculate and display statistics on reconstruction quality
        print("\nReconstruction Quality Statistics:")
        reconstructed_np = reconstructed_points.cpu().numpy()
        
        # Calculate distances from each reconstructed point to the line
        # Using formula: distance = ||(p - a) × (b - a)|| / ||b - a||
        a = np.array(start_point)
        b = np.array(end_point)
        line_vec = b - a
        line_length = np.linalg.norm(line_vec)
        
        # Calculate distances from each reconstructed point to the line
        distances = []
        for p in reconstructed_np:
            # Vector from start to point
            ap = p - a
            # Cross product with line vector
            cross = np.cross(ap, line_vec)
            # Distance = |cross| / |line_vec|
            distance = np.linalg.norm(cross) / line_length
            distances.append(distance)
        
        distances = np.array(distances)
        print(f"Mean distance from original line: {np.mean(distances):.3f} voxels")
        print(f"Max distance from original line: {np.max(distances):.3f} voxels")
        print(f"Min distance from original line: {np.min(distances):.3f} voxels")
        print(f"Standard deviation of distances: {np.std(distances):.3f} voxels")
        
        # Calculate how much of the original line was reconstructed
        # Project reconstructed points onto the line to see coverage
        line_dir = line_vec / line_length
        projections_onto_line = []
        for p in reconstructed_np:
            # Vector from start to point
            ap = p - a
            # Projection onto line = dot(ap, line_dir)
            proj = np.dot(ap, line_dir)
            # Normalize to [0, 1] range
            norm_proj = proj / line_length
            projections_onto_line.append(norm_proj)
        
        projections_onto_line = np.array(projections_onto_line)
        min_proj = np.min(projections_onto_line)
        max_proj = np.max(projections_onto_line)
        coverage = max_proj - min_proj
        print(f"Line coverage: {coverage * 100:.1f}% (from position {min_proj:.3f} to {max_proj:.3f})")
    else:
        print("No points were reconstructed.")
    
    # 5. Reconstruct 3D volume from projections
    print("\n5. Reconstructing 3D volume from projections...")
    start_time = time()
    
    # Use sparse reconstruction
    reconstructed_sparse_volume = reconstructor.reconstruct_sparse_volume(projections, threshold=0.1, voxel_size=1.0, fast_merge=fast_merge)
    reconstructed_coords, reconstructed_values, _ = reconstructed_sparse_volume
    
    # Convert to dense for visualization
    reconstructed_volume = torch.zeros(volume_shape, device=device)
    if reconstructed_coords.shape[0] > 0:
        reconstructed_volume[reconstructed_coords[:, 0], reconstructed_coords[:, 1], reconstructed_coords[:, 2]] = reconstructed_values
    
    print(f"Reconstructed volume in {time() - start_time:.2f} seconds")
    print(f"Reconstructed {reconstructed_coords.shape[0]} voxels")
    
    # Visualize the reconstructed volume
    fig = visualize_volume(reconstructed_volume)
    fig.savefig('line_reconstructed_volume.png')
    plt.close(fig)
    
    # 6. Evaluate the reconstruction
    print("\n6. Evaluating reconstruction...")
    metrics = reconstructor.evaluate_reconstruction(dense_volume, reconstructed_volume, threshold=0.1)
    print("Evaluation metrics:")
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    print("\n=== Line example completed ===")
    print("Generated images:")
    print("  - line_volume.png")
    print("  - line_projections.png")
    print("  - line_reconstruction.png")
    print("  - line_reconstructed_volume.png")

if __name__ == "__main__":
    # Set to True for detailed debug output, False for concise output
    debug_mode = True
    # Set to True for maximum speed, False for maximum precision
    fast_merge_mode = True
    run_line_example(debug=debug_mode, fast_merge=fast_merge_mode) 