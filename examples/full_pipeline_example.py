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

def create_synthetic_volume(size=100, num_points=5, point_size=3.0, device='cuda'):
    """
    Create a synthetic 3D volume with randomly placed points.
    
    Args:
        size (int): Size of the volume (size x size x size)
        num_points (int): Number of random points to generate
        point_size (float): Size of the points (standard deviation of Gaussian)
        device (str): Device to use ('cuda' or 'cpu')
        
    Returns:
        tuple: (volume, points) - The volume tensor and the original 3D points
    """
    # Create an empty volume
    volume = torch.zeros((size, size, size), device=device)
    
    # Generate random 3D points
    points = torch.randint(
        int(point_size * 3), 
        size - int(point_size * 3), 
        (num_points, 3),
        device=device
    ).float()
    
    # Place points in the volume
    for point in points:
        x, y, z = point.long()
        # Create a Gaussian blob at each point
        for dx in range(-int(3*point_size), int(3*point_size) + 1):
            for dy in range(-int(3*point_size), int(3*point_size) + 1):
                for dz in range(-int(3*point_size), int(3*point_size) + 1):
                    nx, ny, nz = x + dx, y + dy, z + dz
                    if 0 <= nx < size and 0 <= ny < size and 0 <= nz < size:
                        dist = torch.sqrt(torch.tensor(dx**2 + dy**2 + dz**2, device=device))
                        value = torch.exp(-(dist/point_size)**2 / 2)
                        volume[nx, ny, nz] = max(volume[nx, ny, nz], value)
    
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

def run_full_pipeline():
    """
    Run the full pipeline from creating a synthetic volume to reconstruction and evaluation.
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
    
    # 1. Create a synthetic volume
    print("\n1. Creating synthetic volume...")
    start_time = time()
    volume, original_points = create_synthetic_volume(
        size=volume_size,
        num_points=num_points,
        point_size=point_size,
        device=device
    )
    print(f"Created volume with {num_points} points in {time() - start_time:.2f} seconds")
    print(f"Original points:\n{original_points.cpu().numpy()}")
    
    # Visualize the volume
    fig = visualize_volume(volume)
    fig.savefig('original_volume.png')
    plt.close(fig)
    
    # 2. Create the reconstructor
    print("\n2. Creating reconstructor...")
    reconstructor = LArTPCReconstructor(
        volume_shape=(volume_size, volume_size, volume_size),
        tolerance=tolerance,
        merge_tolerance=merge_tolerance,
        device=device
    )
    
    # 3. Project the volume to 2D projections
    print("\n3. Projecting volume to 2D...")
    start_time = time()
    projections = reconstructor.project_volume(volume)
    print(f"Projected volume in {time() - start_time:.2f} seconds")
    
    # Visualize the projections
    fig = visualize_projections(projections)
    fig.savefig('projections.png')
    plt.close(fig)
    
    # 4. Reconstruct 3D points from projections
    print("\n4. Reconstructing 3D points from projections...")
    start_time = time()
    reconstructed_points = reconstructor.reconstruct_from_projections(projections, threshold=0.1)
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
    
    # 5. Reconstruct 3D volume from projections
    print("\n5. Reconstructing 3D volume from projections...")
    start_time = time()
    reconstructed_volume = reconstructor.reconstruct_volume(projections, threshold=0.1, voxel_size=1.0)
    print(f"Reconstructed volume in {time() - start_time:.2f} seconds")
    
    # Visualize the reconstructed volume
    fig = visualize_volume(reconstructed_volume)
    fig.savefig('reconstructed_volume.png')
    plt.close(fig)
    
    # 6. Evaluate the reconstruction
    print("\n6. Evaluating reconstruction...")
    metrics = reconstructor.evaluate_reconstruction(volume, reconstructed_volume, threshold=0.1)
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
    run_full_pipeline() 