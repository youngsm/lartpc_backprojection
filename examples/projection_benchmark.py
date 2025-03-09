import torch
import numpy as np
import time
import os
import sys

# Add the parent directory to the path to import the package
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from lartpc_reconstruction import LArTPCReconstructor

def create_test_sparse_volume(shape=(100, 100, 100), num_points=10000, device='cuda'):
    """Create a sparse test volume with random points."""
    # Random coordinates within the volume
    coords = torch.randint(0, min(shape), (num_points, 3), device=device)
    
    # Random values for the points
    values = torch.rand(num_points, device=device)
    
    return coords, values, shape

def create_test_dense_volume(shape=(100, 100, 100), sparsity=0.01, device='cuda'):
    """Create a dense test volume with specified sparsity."""
    # Create an empty volume
    volume = torch.zeros(shape, device=device)
    
    # Calculate how many points to set
    total_voxels = volume.numel()
    num_nonzero = int(total_voxels * sparsity)
    
    # Generate random indices
    indices = torch.randperm(total_voxels, device=device)[:num_nonzero]
    
    # Convert flat indices to 3D coordinates
    x = indices // (shape[1] * shape[2])
    y = (indices % (shape[1] * shape[2])) // shape[2]
    z = indices % shape[2]
    
    # Set random values
    volume[x, y, z] = torch.rand(num_nonzero, device=device)
    
    return volume

def benchmark_projection_functions():
    """Benchmark the performance of different projection functions."""
    print("Benchmarking projection functions...")
    
    # Use CUDA if available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Create LArTPCReconstructor
    volume_shape = (100, 100, 100)
    plane_angles = {0: 0.0, 1: np.pi/3, 2: 2*np.pi/3}
    reconstructor = LArTPCReconstructor(
        volume_shape=volume_shape, 
        device=device,
        plane_angles=plane_angles,
        debug=False  # Turn off debug for fair timing
    )
    
    # Create test volumes
    print("\nCreating test volumes...")
    sparse_volumes = [
        create_test_sparse_volume(volume_shape, num_points=1000, device=device),
        create_test_sparse_volume(volume_shape, num_points=10000, device=device),
        create_test_sparse_volume(volume_shape, num_points=100000, device=device)
    ]
    
    dense_volumes = [
        create_test_dense_volume(volume_shape, sparsity=0.001, device=device),
        create_test_dense_volume(volume_shape, sparsity=0.01, device=device),
        create_test_dense_volume(volume_shape, sparsity=0.1, device=device)
    ]
    
    # Test all volumes
    all_volumes = [
        ("Sparse (1,000 points)", sparse_volumes[0]),
        ("Sparse (10,000 points)", sparse_volumes[1]),
        ("Sparse (100,000 points)", sparse_volumes[2]),
        ("Dense (0.1% filled)", dense_volumes[0]),
        ("Dense (1% filled)", dense_volumes[1]),
        ("Dense (10% filled)", dense_volumes[2])
    ]
    
    # Run benchmarks
    print("\nRunning benchmarks...")
    results = []
    
    for volume_name, volume in all_volumes:
        print(f"\nBenchmarking {volume_name}...")
        
        # Benchmark standard projection
        start_time = time.time()
        standard_projections = reconstructor.project_volume(volume)
        standard_time = time.time() - start_time
        print(f"  Standard projection time: {standard_time:.4f} seconds")
        
        # Benchmark differentiable projection
        start_time = time.time()
        diff_projections = reconstructor.project_volume_differentiable(volume)
        diff_time = time.time() - start_time
        print(f"  Differentiable projection time: {diff_time:.4f} seconds")
        
        # Store results
        results.append({
            'name': volume_name,
            'standard_time': standard_time,
            'differentiable_time': diff_time,
            'speedup': standard_time / diff_time if diff_time > 0 else float('inf')
        })
        
        # Verify that the projections match
        for plane_id in reconstructor.solver.plane_angles:
            std_proj = standard_projections[plane_id]
            diff_proj = diff_projections[plane_id]
            
            # Check shapes
            print(f"  Plane {plane_id}: Standard shape {std_proj.shape}, Differentiable shape {diff_proj.shape}")
            
            # Calculate the difference (allowing for small numerical differences)
            if std_proj.shape == diff_proj.shape:
                max_diff = torch.max(torch.abs(std_proj - diff_proj)).item()
                print(f"  Maximum difference: {max_diff:.6f}")
            else:
                print("  Shapes don't match, can't compare values")
    
    # Print summary
    print("\nBenchmark Summary:")
    print("-" * 80)
    print(f"{'Volume Type':<20} {'Standard (s)':<15} {'Differentiable (s)':<20} {'Speedup':<10}")
    print("-" * 80)
    
    for result in results:
        print(f"{result['name']:<20} {result['standard_time']:<15.4f} {result['differentiable_time']:<20.4f} {result['speedup']:<10.2f}x")
    
    print("-" * 80)

if __name__ == "__main__":
    benchmark_projection_functions() 