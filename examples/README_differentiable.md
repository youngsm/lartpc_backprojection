# Differentiable LArTPC Reconstruction

This extension to the LArTPC reconstruction library adds differentiable operations that enable gradient-based optimization for 3D reconstruction.

## Overview

Traditional LArTPC reconstruction uses geometric methods to find intersections between backprojected lines from 2D projections. While effective, this approach doesn't easily allow for optimization or learning. By making key operations differentiable, we can:

1. Use gradient-based optimization to improve reconstructions
2. Incorporate reconstruction into deep learning pipelines
3. Automatically tune parameters like thresholds and tolerances 
4. Learn optimal reconstruction directly from data

## Implementation Details

### Key Components

1. **Differentiable Projection**: 
   - `project_volume_differentiable`: Dense volume projection with gradient propagation
   - `project_sparse_volume_differentiable`: Sparse volume projection with gradient propagation

2. **Optimization Techniques**:
   - Instead of hard rounding, we use bilinear interpolation to maintain differentiability
   - The scatter_add_ operation preserves gradients for accumulation
   - All angle calculations (sin/cos) are performed with gradient tracking
   
3. **Sparse Point-Based Optimization**:
   - New `optimize_sparse_point_intensities` method for fixed-position point optimization
   - Only optimizes intensity values (alpha) for backprojected points
   - Periodically prunes low-intensity points to maintain sparsity

### Technical Improvements

- **Smooth Approximations**: For operations like thresholding, we use differentiable approximations
- **Vectorized Operations**: Optimized for parallel execution on GPU
- **Memory Efficiency**: Handling large volumes through batched processing
- **Sparse Reconstruction**: Optimizes only a sparse set of points rather than a dense volume

## Example Usage

### Dense Volume Optimization

The `differentiable_reconstruction.py` example demonstrates using the differentiable pipeline for dense volume optimization:

```python
# Create reconstructor
reconstructor = LArTPCReconstructor(...)

# Generate projections from known volume
projections = reconstructor.project_volume_differentiable(ground_truth)

# Create an initial volume to optimize
initial_volume = torch.rand(volume_shape, device=device)
initial_volume.requires_grad_(True)  # Enable gradients

# Create optimizer
optimizer = optim.Adam([initial_volume], lr=0.01)

# Optimization loop
for i in range(num_iterations):
    # Forward pass: project current volume
    current_projections = reconstructor.project_volume_differentiable(initial_volume)
    
    # Calculate loss
    loss = 0
    for plane_id in projections:
        loss += torch.mean((projections[plane_id] - current_projections[plane_id])**2)
    
    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

### Sparse Point-Based Optimization

The `sparse_differentiable_reconstruction.py` and `realistic_sparse_reconstruction.py` examples demonstrate the more efficient point-based approach:

```python
# Generate candidate points from backprojection
candidate_points = reconstructor.reconstruct_from_projections(
    thresholded_projections, 
    threshold=threshold,
    fast_merge=True, 
    snap_to_grid=True
)

# Optimize only the intensities of these fixed-position points
optimized_coords, optimized_values, loss_history, num_points = reconstructor.optimize_sparse_point_intensities(
    candidate_points=candidate_points,
    target_projections=projections,
    num_iterations=300,
    lr=0.01,
    pruning_threshold=0.05,
    pruning_interval=100,
    l1_weight=0.01
)
```

## Advantages of Sparse Point-Based Optimization

The sparse point-based approach offers several advantages:

1. **Memory Efficiency**: Only stores and processes non-zero points
2. **Computation Speed**: Much faster than dense volume optimization
3. **Better Regularization**: Natural sparsity through point pruning
4. **Fixed Positions**: Uses the geometrically accurate positions from backprojection
5. **Focused Learning**: Learns only intensity values, simplifying the optimization problem

## Future Directions

1. **Differentiable Backprojection**: Make the backprojection operation differentiable
2. **End-to-End Optimization**: Create a fully differentiable pipeline from raw detector signals to 3D reconstruction
3. **Physics-Informed Neural Networks**: Incorporate physical constraints into the optimization
4. **Noise Modeling**: Add learnable noise models to improve robustness
5. **Dynamic Point Positions**: Allow for small adjustments to point positions during optimization

## References

- [PyTorch Autograd Documentation](https://pytorch.org/docs/stable/autograd.html)
- [Differentiable Volume Rendering: Learning Implicit 3D Representations without 3D Supervision](https://arxiv.org/abs/1912.07372)
- [Physics-Informed Neural Networks](https://arxiv.org/abs/1711.10561) 