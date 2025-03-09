import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import sys

# Add the parent directory to the path to import the package
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lartpc_reconstruction import LineIntersectionSolver

def create_synthetic_volume(size=100, num_points=5, device='cuda'):
    """
    Create a synthetic 3D volume with randomly placed points.
    
    Args:
        size (int): Size of the volume (size x size x size)
        num_points (int): Number of random points to generate
        device (str): Device to use ('cuda' or 'cpu')
        
    Returns:
        tuple: (volume, points) - The volume tensor and the original 3D points
    """
    # Create an empty volume
    volume = torch.zeros((size, size, size), device=device)
    
    # Generate random 3D points
    points = torch.randint(10, size - 10, (num_points, 3), device=device).float()
    
    # Place points in the volume
    for point in points:
        x, y, z = point.long()
        # Create a small Gaussian-like blob at each point
        for dx in range(-2, 3):
            for dy in range(-2, 3):
                for dz in range(-2, 3):
                    nx, ny, nz = x + dx, y + dy, z + dz
                    if 0 <= nx < size and 0 <= ny < size and 0 <= nz < size:
                        dist = torch.sqrt(torch.tensor(dx**2 + dy**2 + dz**2))
                        value = torch.exp(-dist)
                        volume[nx, ny, nz] = value
    
    return volume, points

def project_and_reconstruct():
    """
    Example of projecting a 3D volume to 2D projections
    and then reconstructing the 3D points from the projections.
    """
    print("Creating synthetic volume...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    volume_size = 100
    volume, original_points = create_synthetic_volume(size=volume_size, num_points=5, device=device)
    
    print(f"Using device: {device}")
    print(f"Original points:\n{original_points.cpu().numpy()}")
    
    # Create the solver
    solver = LineIntersectionSolver(
        volume_shape=(volume_size, volume_size, volume_size),
        tolerance=1.0,
        merge_tolerance=2.0,
        device=device
    )
    
    # Create projections for each plane
    projections = {}
    plane_ids = [0, 1, 2]  # Vertical, horizontal, and 60-degree planes
    
    print("Computing projections...")
    for plane_id in plane_ids:
        theta = solver.plane_angles[plane_id]
        u_min = solver.u_min_values[plane_id]
        projection = solver.project_volume_cuda(volume, theta, u_min, device)
        projections[plane_id] = projection
    
    # Apply a threshold to the projections to create binary hit maps
    for plane_id in plane_ids:
        projections[plane_id] = (projections[plane_id] > 0.1).float()
    
    print("Solving inverse problem...")
    # Solve the inverse problem
    reconstructed_points = solver.solve_inverse_problem(projections)
    
    print(f"Reconstructed {reconstructed_points.shape[0]} points.")
    print(f"Reconstructed points:\n{reconstructed_points.cpu().numpy()}")
    
    # Visualize the results
    visualize_results(original_points.cpu().numpy(), reconstructed_points.cpu().numpy(), volume_size)

def direct_line_intersection_example():
    """
    Example of directly creating lines and finding their intersections.
    """
    print("Direct line intersection example...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    volume_size = 100
    
    # Create the solver
    solver = LineIntersectionSolver(
        volume_shape=(volume_size, volume_size, volume_size),
        tolerance=1.0,
        merge_tolerance=2.0,
        device=device
    )
    
    # Create lines from different planes
    # Each line is defined by a point and a direction vector
    lines_by_plane = {
        0: [],  # Vertical plane
        1: [],  # Horizontal plane
        2: []   # 60-degree plane
    }
    
    # Points where the lines should intersect
    target_points = np.array([
        [30, 40, 50],
        [60, 70, 20],
        [80, 30, 60]
    ], dtype=np.float32)
    
    print(f"Target intersection points:\n{target_points}")
    
    # For each target point, create lines from each plane that pass through or near the point
    for point in target_points:
        for plane_id in [0, 1, 2]:
            theta = solver.plane_angles[plane_id]
            
            # Create a direction vector perpendicular to the wire
            direction = np.array([0, np.cos(theta), np.sin(theta)], dtype=np.float32)
            
            # Add a small random perturbation to simulate detector resolution
            perturb = np.random.normal(0, 0.5, 3).astype(np.float32)
            start_point = point + perturb
            
            # Create the line
            from lartpc_reconstruction import Line3D
            line = Line3D(start_point, direction, plane_id)
            lines_by_plane[plane_id].append(line)
    
    # Find intersections using CPU implementation
    print("Finding intersections using CPU...")
    cpu_intersections = solver.intersect_lines_cpu(lines_by_plane)
    print(f"Found {len(cpu_intersections)} intersections on CPU.")
    print(f"CPU intersections:\n{cpu_intersections}")
    
    # Convert to tensors for GPU
    points_tensors = {}
    directions_tensors = {}
    plane_ids_tensors = {}
    
    for plane_id, lines in lines_by_plane.items():
        from lartpc_reconstruction.cuda_kernels import get_line_parameters_tensor
        points, directions, plane_ids = get_line_parameters_tensor(lines, device=device)
        points_tensors[plane_id] = points
        directions_tensors[plane_id] = directions
        plane_ids_tensors[plane_id] = plane_ids
    
    # Find intersections using GPU
    print("Finding intersections using GPU...")
    all_points = torch.cat([points_tensors[i] for i in range(3)], dim=0)
    all_directions = torch.cat([directions_tensors[i] for i in range(3)], dim=0)
    all_plane_ids = torch.cat([plane_ids_tensors[i] for i in range(3)], dim=0)
    
    # Find intersections between planes
    intersection_points = []
    intersection_distances = []
    
    for i in range(3):
        for j in range(i + 1, 3):
            mask1 = all_plane_ids == i
            mask2 = all_plane_ids == j
            
            points1 = all_points[mask1]
            directions1 = all_directions[mask1]
            plane_ids1 = all_plane_ids[mask1]
            
            points2 = all_points[mask2]
            directions2 = all_directions[mask2]
            plane_ids2 = all_plane_ids[mask2]
            
            from lartpc_reconstruction.cuda_kernels import find_intersections_cuda
            points, indices1, indices2, distances = find_intersections_cuda(
                points1, directions1, plane_ids1,
                points2, directions2, plane_ids2,
                tolerance=1.0, device=device
            )
            
            intersection_points.append(points)
            intersection_distances.append(distances)
    
    # Concatenate all intersection points
    if intersection_points:
        all_intersection_points = torch.cat(intersection_points, dim=0)
        all_distances = torch.cat(intersection_distances, dim=0)
        
        from lartpc_reconstruction.cuda_kernels import merge_nearby_intersections_cuda
        merged_points = merge_nearby_intersections_cuda(
            all_intersection_points,
            all_distances,
            tolerance=2.0
        )
        
        print(f"Found {merged_points.shape[0]} intersections on GPU.")
        print(f"GPU intersections:\n{merged_points.cpu().numpy()}")
        
        # Visualize the results
        visualize_results(target_points, merged_points.cpu().numpy(), volume_size)
    else:
        print("No intersections found on GPU.")

def visualize_results(original_points, reconstructed_points, volume_size):
    """
    Visualize the original and reconstructed points in 3D.
    
    Args:
        original_points (np.ndarray): Original 3D points
        reconstructed_points (np.ndarray): Reconstructed 3D points
        volume_size (int): Size of the volume
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot original points
    ax.scatter(original_points[:, 0], original_points[:, 1], original_points[:, 2], 
               color='blue', label='Original', s=100, marker='o')
    
    # Plot reconstructed points
    ax.scatter(reconstructed_points[:, 0], reconstructed_points[:, 1], reconstructed_points[:, 2], 
               color='red', label='Reconstructed', s=50, marker='x')
    
    # Set axis limits
    ax.set_xlim(0, volume_size)
    ax.set_ylim(0, volume_size)
    ax.set_zlim(0, volume_size)
    
    # Add labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Original vs Reconstructed Points')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig('img/original_vs_reconstructed.png')

if __name__ == "__main__":
    # Choose which example to run
    example_mode = "direct"  # "projection" or "direct"
    
    if example_mode == "projection":
        project_and_reconstruct()
    else:
        direct_line_intersection_example() 