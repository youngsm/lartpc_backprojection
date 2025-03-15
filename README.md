# LArTPC Backprojection Project

This repository provides a barebones differentiable 2D <--> 3D forward and backward simulator for LArTPC wire plane projections. It written using Pytorch, and is designed for extremely fast and scalable backprojection and optimization of 3D images using just 2D projection images.

It is divergent from current methods, which use neural network-based priors to reconstruct images, by leveraging the implicit geometric constraints built into the 2D projections read out by wire-based LArTPCs.

It is extremely simple, and makes many nonphysical assumptions: no charge diffusion or attenuation, and assumed charges can only activate 1 wire per plane. As such, this is a proof of concept/starting point for this method more than anything.

----


There are three main functions that perform the bulk of the work; all other classes methods are scaffolding that use these. They are found in `cuda_kernels.py`, and are:

* `project_coordinates_to_plane_exact`: projects a set of 3D coordinates and values to a 2D projection. (i.e., simulates the LArTPC readout after deconvolution).

* `backproject_hits_into_lines`: given a 2D projection, will create a set of line representations that correspond to each hit wire.

* `find_intersections_between_lines`: given two sets of line representations corresponding to lines from two 2D projections, finds all points of intersection between them. This includes doublet (two line) intersections and triplet (three line) intersections.

## Backprojection Example

`intersection_solver.py` incorporates the backprojection operation by combining the latter two functions. It has the `LineIntersectionSolver` method, which is used as follows:

```python
>>> solver = LineIntersectionSolver(
    volume_shape=(100,100,100),
    plane_angles={0: np.pi/2, 1: np.pi/6, 2: -np.pi/6}
)
>>> solver.solve(
    {0: projection0, 1: projection1, 2: projection2},
    snap_to_grid=True,
    voxel_size=1.0,
)
```

where `projectionN` is a 100x100 image containing projection values along the X and U axes. 


## Full Reconstruction Example

Full reconstruction is done with the `LArTPCReconstructor`, which is found in `reconstructor.py`. 

```python
>>> reconstructor = LArTPCReconstructor(
    volume_shape=(100,100,100),
    plane_angles={0: np.pi/2, 1: np.pi/6, -np.pi/6}
)
>>> final_coords, energy_values = reconstructor.reconstruct_points_from_projections(
    {0: true_projection0, 1: true_projection1, 2: true_projection2},
    snap_to_grid=True,
    voxel_size=1.0,
    lr=0.01,
    num_iterations=2000,
    pruning_interval=200,
    warmup_iterations=500,
    pruning_threshold=1e-3,
    loss_func='l2',
    l1_weight=0.01
)
```

or, given the candidate point array and target projections,

```python
>>> final_coords, energy_values, *__ = reconstructor.optimize_point_intensities(
    candidate_points,
    {0: true_projection0, 1: true_projection1, 2: true_projection2},
    lr=0.01,
    num_iterations=2000,
    num_iterations=2000,
    pruning_interval=200,
    warmup_iterations=500,
    pruning_threshold=1e-3,
    loss_func='l2',
    l1_weight=0.01
)
```

### Forward projection

The reconstructor also has a method to perform a differentiable forward projection from 3D points to 2D projections. Assuming you have initialized a reconstructor like above, you can do:

```python
>>> volume_shape = (100,100,100)
>>> reconstructor.project_sparse_volume(
    (coordinates, values, volume_shape)
)
```

this will return a dictionary containing each projection.

### Example code

The `examples/` directory contains a couple scripts I used to evaluate this algorithm. The notebook contains code I used in the early stages of this project, and will probably not run. `realistic_sparse_reconstruction.py` is a script that runs full reconstruction on a single LArTPC image, and print out helpful evaluation metrics and saves pretty plots to look at. `line_visualization.py` attempts to visualize the wire planes but it's very finicky.

### Dataset

The dataset used for this project is the PILArNet-M dataset, which can be found at this [Github link](https://github.com/DeepLearnPhysics/PoLAr-MAE/blob/main/DATASET.md).
___

### Contact

Sam Young, youngsam@stanford.edu, https://youngsm.com/