import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from time import time

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

def create_sparse_synthetic_volume(size=100, num_points=5, point_size=3.0, device='cuda'):
    """
    Create a synthetic 3D volume with randomly placed points, using a sparse approach.
    
    Args:
        size (int): Size of the volume (size x size x size)
        num_points (int): Number of random points to generate
        point_size (float): Size of the points (standard deviation of Gaussian)
        device (str): Device to use ('cuda' or 'cpu')
        
    Returns:
        tuple: (sparse_volume, points) - The sparse volume representation and the original 3D points
               sparse_volume is a tuple of (coords, values, shape)
    """
    # Generate random 3D points
    points = torch.randint(
        int(point_size * 3), 
        size - int(point_size * 3), 
        (num_points, 3),
        device=device
    ).float()
    
    # Preallocate the coords and values lists for non-zero voxels
    max_voxels = num_points * int((6*point_size)**3)  # Conservative upper bound
    coords_list = []
    values_list = []
    
    # Create point clouds around each point
    for point in points:
        x, y, z = point.long()
        # Calculate the ranges for the Gaussian blob
        x_range = torch.arange(max(0, x - int(3*point_size)), min(size, x + int(3*point_size) + 1), device=device)
        y_range = torch.arange(max(0, y - int(3*point_size)), min(size, y + int(3*point_size) + 1), device=device)
        z_range = torch.arange(max(0, z - int(3*point_size)), min(size, z + int(3*point_size) + 1), device=device)
        
        # Create meshgrid for the blob
        xx, yy, zz = torch.meshgrid(x_range, y_range, z_range, indexing='ij')
        
        # Calculate distances
        distances = torch.sqrt((xx - x.float())**2 + (yy - y.float())**2 + (zz - z.float())**2)
        
        # Calculate values with Gaussian
        values = torch.exp(-(distances/point_size)**2 / 2)
        
        # Filter out low values for sparsity
        mask = values > 0.01
        if mask.sum() > 0:
            # Get coordinates and values for this point's blob
            point_coords = torch.stack([xx[mask], yy[mask], zz[mask]], dim=1)
            point_values = values[mask]
            
            coords_list.append(point_coords)
            values_list.append(point_values)
    
    # Combine all points
    if coords_list:
        all_coords = torch.cat(coords_list, dim=0)
        all_values = torch.cat(values_list, dim=0)
        
        # Handle duplicate coordinates by taking max value
        # First, create a unique identifier for each coordinate
        coord_ids = all_coords[:, 0] * size**2 + all_coords[:, 1] * size + all_coords[:, 2]
        
        # Find unique coordinates and their indices
        unique_ids, inverse_indices = torch.unique(coord_ids, return_inverse=True)
        
        # Create new coordinates and values arrays
        unique_coords = torch.zeros((len(unique_ids), 3), dtype=torch.long, device=device)
        unique_values = torch.zeros(len(unique_ids), device=device)
        
        # For each unique coordinate, find all points with that coordinate
        # and take the maximum value
        for i, unique_id in enumerate(unique_ids):
            mask = (coord_ids == unique_id)
            unique_coords[i] = all_coords[mask][0]  # All coords with this ID are the same
            unique_values[i] = torch.max(all_values[mask])
    else:
        # Handle case with no points (unlikely but possible)
        unique_coords = torch.zeros((0, 3), dtype=torch.long, device=device)
        unique_values = torch.zeros(0, device=device)
    
    return (unique_coords, unique_values, (size, size, size)), points

def create_synthetic_volume(size=100, num_points=5, point_size=3.0, device='cuda'):
    """
    Create a synthetic 3D volume with randomly placed points.
    Now using the sparse implementation internally for efficiency,
    but still returns a dense volume for compatibility.
    
    Args:
        size (int): Size of the volume (size x size x size)
        num_points (int): Number of random points to generate
        point_size (float): Size of the points (standard deviation of Gaussian)
        device (str): Device to use ('cuda' or 'cpu')
        
    Returns:
        tuple: (volume, points) - The volume tensor and the original 3D points
    """
    # Create sparse volume
    sparse_volume, points = create_sparse_synthetic_volume(size, num_points, point_size, device)
    coords, values, shape = sparse_volume
    
    # Convert to dense volume
    volume = torch.zeros(shape, device=device)
    if coords.shape[0] > 0:
        volume[coords[:, 0], coords[:, 1], coords[:, 2]] = values
    
    return volume, points

def create_synthetic_lines(points, angles, noise=1.0, device='cuda'):
    """
    Create synthetic lines that pass through the given points with some noise.
    
    Args:
        points (torch.Tensor): Points to create lines through (N, 3)
        angles (dict): Dictionary mapping plane_id to angle in radians
        noise (float): Amount of noise to add to the lines
        device (str): Device to use ('cuda' or 'cpu')
        
    Returns:
        dict: Dictionary mapping plane_id to lists of Line3D objects
    """
    # Convert points to numpy if they're torch tensors
    if isinstance(points, torch.Tensor):
        points = points.cpu().numpy()
    
    # Create lines for each plane
    lines_by_plane = {}
    
    for plane_id, theta in angles.items():
        lines = []
        # Direction perpendicular to the wire
        direction = np.array([0.0, np.cos(theta), np.sin(theta)])
        
        # For each point, create a line that passes through it
        for point in points:
            # Add noise to the point
            noisy_point = point + np.random.normal(0, noise, 3)
            
            # Create the line
            line = Line3D(noisy_point, direction, plane_id)
            lines.append(line)
        
        lines_by_plane[plane_id] = lines
    
    return lines_by_plane

def run_full_pipeline(debug=True, fast_merge=True):
    """
    Run the full pipeline from creating a synthetic volume to reconstruction and evaluation.
    
    Args:
        debug (bool): Whether to print detailed debug information
        fast_merge (bool): Whether to use fast merging mode for intersection clustering
    """
    print("=== Running full pipeline example ===")
    
    # Check if CUDA is available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Set parameters
    volume_size = 100
    num_points = 5
    point_size = 3.0
    tolerance = 2.0
    merge_tolerance = 3.0
    
    # 1. Create a synthetic volume (using the sparse approach internally)
    print("\n1. Creating synthetic volume...")
    start_time = time()
    
    # Use sparse representation for efficiency
    sparse_volume, original_points = create_sparse_synthetic_volume(
        size=volume_size,
        num_points=num_points,
        point_size=point_size,
        device=device
    )
    
    # For visualization, create a dense volume only when needed
    coords, values, shape = sparse_volume
    
    print(f"Created sparse volume with {coords.shape[0]} non-zero voxels in {time() - start_time:.2f} seconds")
    print(f"Original points:\n{original_points.cpu().numpy()}")
    
    # Convert to dense for visualization only
    print("Converting to dense for visualization...")
    dense_volume = torch.zeros(shape, device=device)
    if coords.shape[0] > 0:
        dense_volume[coords[:, 0], coords[:, 1], coords[:, 2]] = values
    
    # Visualize the volume
    fig = visualize_volume(dense_volume)
    fig.savefig('original_volume.png')
    plt.close(fig)
    
    # 2. Create the reconstructor
    print("\n2. Creating reconstructor...")
    reconstructor = LArTPCReconstructor(
        volume_shape=(volume_size, volume_size, volume_size),
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
    fig.savefig('projections.png')
    plt.close(fig)
    
    # 4. Reconstruct 3D points from projections
    print("\n4. Reconstructing 3D points from projections...")
    start_time = time()
    reconstructed_points = reconstructor.reconstruct_from_projections(projections, threshold=0.1, fast_merge=fast_merge)
    print(f"Reconstructed {reconstructed_points.size(0)} points in {time() - start_time:.2f} seconds")
    print(f"Reconstructed points:\n{reconstructed_points.cpu().numpy()}")
    
    # Visualize original vs reconstructed points
    fig = visualize_original_vs_reconstructed(
        original_points.cpu().numpy(),
        reconstructed_points.cpu().numpy(),
        volume_shape=(volume_size, volume_size, volume_size)
    )
    fig.savefig('original_vs_reconstructed.png')
    plt.close(fig)
    
    # 5. Reconstruct 3D volume from projections (use sparse reconstruction for efficiency)
    print("\n5. Reconstructing 3D volume from projections...")
    start_time = time()
    
    # Use sparse reconstruction
    reconstructed_sparse_volume = reconstructor.reconstruct_sparse_volume(projections, threshold=0.1, voxel_size=1.0, fast_merge=fast_merge)
    reconstructed_coords, reconstructed_values, _ = reconstructed_sparse_volume
    
    # Convert to dense for visualization
    reconstructed_volume = torch.zeros(shape, device=device)
    if reconstructed_coords.shape[0] > 0:
        reconstructed_volume[reconstructed_coords[:, 0], reconstructed_coords[:, 1], reconstructed_coords[:, 2]] = reconstructed_values
    
    print(f"Reconstructed volume in {time() - start_time:.2f} seconds")
    
    # Visualize the reconstructed volume
    fig = visualize_volume(reconstructed_volume)
    fig.savefig('reconstructed_volume.png')
    plt.close(fig)
    
    # 6. Evaluate the reconstruction
    print("\n6. Evaluating reconstruction...")
    metrics = reconstructor.evaluate_reconstruction(dense_volume, reconstructed_volume, threshold=0.1)
    print("Evaluation metrics:")
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    # 7. Direct line intersection (alternative approach)
    print("\n7. Testing direct line intersection approach...")
    start_time = time()
    
    # Create synthetic lines through the original points
    lines_by_plane = create_synthetic_lines(
        original_points,
        reconstructor.solver.plane_angles,
        noise=1.0
    )
    
    # Reconstruct points from lines
    cpu_reconstructed_points = reconstructor.reconstruct_from_lines(lines_by_plane)
    print(f"Reconstructed {len(cpu_reconstructed_points)} points using CPU in {time() - start_time:.2f} seconds")
    
    # Visualize lines and intersections
    fig = visualize_lines_and_intersections(
        lines_by_plane,
        cpu_reconstructed_points,
        volume_shape=(volume_size, volume_size, volume_size)
    )
    fig.savefig('lines_and_intersections.png')
    plt.close(fig)
    
    print("\n=== Pipeline completed successfully ===")
    print("Generated images:")
    print("  - original_volume.png")
    print("  - projections.png")
    print("  - original_vs_reconstructed.png")
    print("  - reconstructed_volume.png")
    print("  - lines_and_intersections.png")

if __name__ == "__main__":
    # Set to True for detailed debug output, False for concise output
    debug_mode = True
    # Set to True for maximum speed, False for maximum precision
    fast_merge_mode = True
    run_full_pipeline(debug=debug_mode, fast_merge=fast_merge_mode) 