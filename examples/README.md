# LArTPC Reconstruction Examples

This directory contains examples that demonstrate how to use the LArTPC reconstruction package for finding intersections of backprojected lines.

## Available Examples

1. **Simple Example** (`simple_example.py`):
   A basic example that demonstrates the core functionality with two modes:
   - Projection mode: Projects a 3D volume to 2D, then reconstructs 3D points
   - Direct mode: Directly creates lines and finds their intersections

2. **Full Pipeline Example** (`full_pipeline_example.py`):
   A comprehensive example that demonstrates the complete pipeline:
   - Creating a synthetic 3D volume
   - Projecting to 2D planes
   - Reconstructing 3D points from projections
   - Reconstructing a full 3D volume
   - Evaluating the reconstruction quality
   - Alternative approach with direct line intersection
   - Visualization at each step

## Running the Examples

To run the examples, make sure you have installed the package. From the project root directory:

```bash
# Install the package in development mode
pip install -e .

# Run the full pipeline example
cd lartpc_reconstruction/examples
python full_pipeline_example.py

# Run the simple example
python simple_example.py
```

The examples will generate visualization images in the current directory.

## Example Output

The full pipeline example will generate the following visualizations:

1. `original_volume.png`: Visualization of the synthetic 3D volume
2. `projections.png`: Visualization of the 2D projections from different wire planes
3. `original_vs_reconstructed.png`: Comparison of original and reconstructed 3D points
4. `reconstructed_volume.png`: Visualization of the reconstructed 3D volume
5. `lines_and_intersections.png`: Visualization of 3D lines and their intersections

## Customization

You can modify the example parameters to experiment with different configurations:

- `volume_size`: Size of the 3D volume
- `num_points`: Number of points in the synthetic volume
- `point_size`: Size of the points (standard deviation of Gaussian)
- `tolerance`: Tolerance for intersection testing
- `merge_tolerance`: Tolerance for merging nearby intersections 