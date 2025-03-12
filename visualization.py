import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.colors import LogNorm
from mpl_toolkits.mplot3d import Axes3D
import torch

def visualize_volume(volume, threshold=0.1, alpha=0.5, figsize=(10, 8)):
    """
    Visualize a 3D volume using a 3D scatter plot.
    
    Args:
        volume (numpy.ndarray or torch.Tensor): 3D volume
        threshold (float): Threshold for considering a voxel as non-zero
        alpha (float): Alpha value for transparency
        figsize (tuple): Figure size
        
    Returns:
        matplotlib.figure.Figure: The figure object
    """
    # Convert to numpy if it's a torch tensor
    if isinstance(volume, torch.Tensor):
        volume = volume.cpu().numpy()
    
    # Create figure
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    
    # Find non-zero voxels
    voxels = np.where(volume > threshold)
    x, y, z = voxels
    values = volume[voxels]
    
    # Normalize values for color mapping
    norm = colors.Normalize(vmin=values.min(), vmax=values.max())
    
    # Plot points
    scatter = ax.scatter(x, y, z, c=values, alpha=alpha, cmap='viridis', norm=norm)
    
    # Add colorbar
    plt.colorbar(scatter, ax=ax, label='Intensity')
    
    # Set labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Volume Visualization')
    
    # Set axis limits
    ax.set_xlim(0, volume.shape[0])
    ax.set_ylim(0, volume.shape[1])
    ax.set_zlim(0, volume.shape[2])
    
    return fig

def visualize_projections(projections, figsize=(15, 5), lognorm=False):
    """
    Visualize 2D projections from different wire planes.
    
    Args:
        projections (dict): Dictionary mapping plane_id to projection data
        figsize (tuple): Figure size
        
    Returns:
        matplotlib.figure.Figure: The figure object
    """
    # Convert to numpy if they're torch tensors
    projections_np = {}
    for plane_id, proj in projections.items():
        if isinstance(proj, torch.Tensor):
            projections_np[plane_id] = proj.cpu().numpy()
        else:
            projections_np[plane_id] = proj
    
    # Create figure
    fig, axes = plt.subplots(1, len(projections), figsize=figsize)
    if len(projections) == 1:
        axes = [axes]
    
    # Map plane_id to angle labels
    
    # Plot each projection
    for i, (plane_id, proj) in enumerate(projections_np.items()):
        # Get angle label
        
        # Plot projection
        im = axes[i].imshow(proj, aspect='auto', cmap='viridis', interpolation='none', norm=LogNorm() if lognorm else None)
        axes[i].set_title(f'Projection {i}')
        axes[i].set_xlabel('X (drift)')
        axes[i].set_ylabel('U (wire)')
        plt.colorbar(im, ax=axes[i])
    
    plt.tight_layout()
    return fig

def visualize_lines_and_intersections(lines_by_plane, intersection_points, volume_shape=(100, 100, 100), figsize=(10, 8)):
    """
    Visualize 3D lines and their intersections.
    
    Args:
        lines_by_plane (dict): Dictionary mapping plane_id to list of Line3D objects
        intersection_points (numpy.ndarray or torch.Tensor): Intersection points (N, 3)
        volume_shape (tuple): Shape of the volume
        figsize (tuple): Figure size
        
    Returns:
        matplotlib.figure.Figure: The figure object
    """
    # Convert to numpy if it's a torch tensor
    if isinstance(intersection_points, torch.Tensor):
        intersection_points = intersection_points.cpu().numpy()
    
    # Create figure
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    
    # Set colors for each plane
    colors = {0: 'r', 1: 'g', 2: 'b', 3: 'c'}
    
    # Plot lines
    for plane_id, lines in lines_by_plane.items():
        color = colors.get(plane_id, 'm')
        
        for line in lines:
            # Get points along the line within the volume
            min_bound, max_bound = line.get_bounds(np.zeros(3), np.array(volume_shape))
            
            # Check if the line intersects the volume
            if np.all(min_bound == 0) and np.all(max_bound == 0):
                continue
            
            # Compute parametric values
            t_min = np.min([np.dot(min_bound - line.point, line.direction), 0])
            t_max = np.max([np.dot(max_bound - line.point, line.direction), 0])
            
            # Create points along the line
            t_values = np.linspace(t_min, t_max, 100)
            points = np.array([line.point_at(t) for t in t_values])
            
            # Plot the line
            ax.plot(points[:, 0], points[:, 1], points[:, 2], color=color, alpha=0.3)
    
    # Plot intersection points
    if len(intersection_points) > 0:
        ax.scatter(
            intersection_points[:, 0],
            intersection_points[:, 1],
            intersection_points[:, 2],
            color='k',
            s=50,
            marker='o'
        )
    
    # Set labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Lines and Intersections')
    
    # Set axis limits
    ax.set_xlim(0, volume_shape[0])
    ax.set_ylim(0, volume_shape[1])
    ax.set_zlim(0, volume_shape[2])
    
    # Add a legend
    legend_elements = []
    for plane_id, color in colors.items():
        if plane_id in lines_by_plane:
            label = f'Plane {plane_id}'
            
            from matplotlib.lines import Line2D
            legend_elements.append(Line2D([0], [0], color=color, lw=2, label=label))
    
    legend_elements.append(Line2D([0], [0], marker='o', color='w', markerfacecolor='k', markersize=8, label='Intersections'))
    
    ax.legend(handles=legend_elements)
    
    return fig

def visualize_original_vs_reconstructed(original_points, reconstructed_points, volume_shape=(100, 100, 100), figsize=(10, 8)):
    """
    Visualize original and reconstructed points.
    
    Args:
        original_points (numpy.ndarray or torch.Tensor): Original points (N, 3)
        reconstructed_points (numpy.ndarray or torch.Tensor): Reconstructed points (M, 3)
        volume_shape (tuple): Shape of the volume
        figsize (tuple): Figure size
        
    Returns:
        matplotlib.figure.Figure: The figure object
    """
    # Convert to numpy if they're torch tensors
    if isinstance(original_points, torch.Tensor):
        original_points = original_points.cpu().numpy()
    if isinstance(reconstructed_points, torch.Tensor):
        reconstructed_points = reconstructed_points.cpu().numpy()
    
    # Create figure
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot original points
    if len(original_points) > 0:
        ax.scatter(
            original_points[:, 0],
            original_points[:, 1],
            original_points[:, 2],
            color='b',
            s=100,
            marker='o',
            label='Original'
        )
    
    # Plot reconstructed points
    if len(reconstructed_points) > 0:
        ax.scatter(
            reconstructed_points[:, 0],
            reconstructed_points[:, 1],
            reconstructed_points[:, 2],
            color='r',
            s=50,
            marker='x',
            alpha=0.1,
            label='Reconstructed'
        )
    
    # Set labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Original vs Reconstructed Points')
    
    # Set axis limits
    ax.set_xlim(0, volume_shape[0])
    ax.set_ylim(0, volume_shape[1])
    ax.set_zlim(0, volume_shape[2])
    
    # Add legend
    ax.legend()
    
    return fig 