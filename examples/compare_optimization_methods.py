#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Comparison of Adam and ADMM Optimization Methods for LArTPC Reconstruction

This script compares the performance of Adam (SGD) and ADMM optimization methods
for 3D image reconstruction from 2D projections in LArTPC detectors.
"""

import os
import sys
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import torch

# Add the parent directory to the path to import the package
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from lartpc_reconstruction import LArTPCReconstructor
from lartpc_reconstruction.visualization import (
    visualize_volume, 
    visualize_projections, 
    visualize_original_vs_reconstructed
)

from realistic_sparse_reconstruction import (
    create_helix,
    create_sine_wave,
    create_volume_from_points
)

def compare_optimization_methods(volume_shape=(100, 100, 100), num_planes=3, 
                                 adam_iterations=1000, admm_iterations=200,
                                 noise_level=0.0, make_plots=True):
    """
    Compare Adam and ADMM optimization methods for LArTPC reconstruction.
    
    Args:
        volume_shape (tuple): Shape of the 3D volume
        num_planes (int): Number of projection planes
        adam_iterations (int): Number of iterations for Adam optimizer
        admm_iterations (int): Number of iterations for ADMM optimizer
        noise_level (float): Level of noise to add to projections
        make_plots (bool): Whether to generate plots
        
    Returns:
        dict: Dictionary containing comparison results
    """
    print(f"Comparing Adam and ADMM optimization methods...")
    
    # Use CUDA if available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Create img directory if it doesn't exist
    os.makedirs('img', exist_ok=True)
    
    # Create the ground truth volume
    print("Creating ground truth volume...")
    
    # Create input points for testing
    input_points = []
    
    # Add some test structures
    helix_points = create_helix(start=(40, 40, 20), radius=20, height=60, turns=3, num_points=500)
    sine_points = create_sine_wave(start=(20, 60, 60), length=80, amplitude=20, periods=2, num_points=500)
    second_sine = create_sine_wave(start=(20, 30, 30), length=80, amplitude=15, periods=3, num_points=500)
    
    # Add to input points
    input_points.extend(helix_points)
    input_points.extend(sine_points)
    input_points.extend(second_sine)
    
    # Convert to tensor
    input_points = torch.tensor(input_points, dtype=torch.float32)
    
    # Create sparse ground truth
    ground_truth_sparse, ground_truth, original_points = create_volume_from_points(
        input_points, volume_shape, radius=0.0, device=device
    )
    print(f"Created ground truth with {original_points.shape[0]} non-zero points")
    
    # Create reconstructor with the specified number of planes
    print(f"Creating reconstructor with {num_planes} planes...")
    
    # Create evenly spaced plane angles
    angle_step = np.pi / num_planes
    plane_angles = {i: i * angle_step for i in range(num_planes)}
    
    reconstructor = LArTPCReconstructor(
        volume_shape=volume_shape, 
        device=device,
        plane_angles=plane_angles,
        debug=False
    )
    
    # Generate projections from ground truth
    print("Generating 2D projections from ground truth...")
    projections = reconstructor.project_sparse_volume_differentiable(ground_truth_sparse)
    
    # Add noise to projections if specified
    if noise_level > 0:
        print(f"Adding {noise_level:.2f} noise to projections...")
        for plane_id in projections:
            noise = torch.randn_like(projections[plane_id]) * noise_level
            projections[plane_id] = torch.clamp(projections[plane_id] + noise, 0, 1)
    
    # Generate candidate points for reconstruction
    print("Reconstructing candidate points using standard method...")
    backprojection_threshold = 0.1
    candidate_points = reconstructor.reconstruct_from_projections(
        projections, 
        threshold=backprojection_threshold,
        fast_merge=True, 
        snap_to_grid=True
    )
    
    print(f"Generated {candidate_points.shape[0]} candidate points from backprojection")
    
    # Convert candidate points to long for indexing
    candidate_points_long = candidate_points.long()
    
    # Timing arrays
    methods = ["Adam", "ADMM"]
    times = []
    metrics_list = []
    volumes = []
    
    # Common optimization parameters
    pruning_threshold = 0.01
    l1_weight = 0.01
    
    # 1. Adam optimization
    print("\n===== Running Adam optimization =====")
    adam_start_time = time.time()
    
    # Adam-specific parameters
    learning_rate = 0.01
    pruning_interval = 5000000
    
    # Run Adam optimization
    adam_coords, adam_values, adam_loss_history, adam_points_history = reconstructor.optimize_sparse_point_intensities(
        candidate_points=candidate_points_long,
        target_projections=projections,
        num_iterations=adam_iterations,
        lr=learning_rate,
        pruning_threshold=pruning_threshold,
        pruning_interval=pruning_interval,
        l1_weight=l1_weight,
    )
    
    adam_end_time = time.time()
    adam_time = adam_end_time - adam_start_time
    times.append(adam_time)
    
    # Convert to dense volume for evaluation
    adam_volume = torch.zeros(volume_shape, device=device)
    adam_volume[adam_coords[:, 0], adam_coords[:, 1], adam_coords[:, 2]] = adam_values
    
    # Calculate metrics
    adam_metrics = reconstructor.evaluate_reconstruction(ground_truth, adam_volume, threshold=0.0)
    metrics_list.append(adam_metrics)
    volumes.append(adam_volume)
    
    # 2. ADMM optimization
    print("\n===== Running ADMM optimization =====")
    admm_start_time = time.time()
    
    # ADMM-specific parameters
    rho = 1.0
    alpha = 1.5
    relaxation = 1.5
    pruning_interval_admm = 20
    
    # Run ADMM optimization
    admm_coords, admm_values, admm_loss_history, admm_points_history = reconstructor.optimize_sparse_point_intensities_admm(
        candidate_points=candidate_points_long,
        target_projections=projections,
        num_iterations=admm_iterations,
        rho=rho,
        alpha=alpha,
        pruning_threshold=pruning_threshold,
        pruning_interval=pruning_interval_admm,
        l1_weight=l1_weight,
        relaxation=relaxation
    )
    
    admm_end_time = time.time()
    admm_time = admm_end_time - admm_start_time
    times.append(admm_time)
    
    # Convert to dense volume for evaluation
    admm_volume = torch.zeros(volume_shape, device=device)
    admm_volume[admm_coords[:, 0], admm_coords[:, 1], admm_coords[:, 2]] = admm_values
    
    # Calculate metrics
    admm_metrics = reconstructor.evaluate_reconstruction(ground_truth, admm_volume, threshold=0.0)
    metrics_list.append(admm_metrics)
    volumes.append(admm_volume)
    
    # Generate comparison results
    results = {
        "times": {methods[i]: times[i] for i in range(len(methods))},
        "metrics": {methods[i]: metrics_list[i] for i in range(len(methods))},
        "points_final": {
            "Adam": adam_coords.shape[0],
            "ADMM": admm_coords.shape[0]
        },
        "loss_history": {
            "Adam": adam_loss_history,
            "ADMM": admm_loss_history
        },
        "points_history": {
            "Adam": adam_points_history,
            "ADMM": admm_points_history
        },
        "volumes": {methods[i]: volumes[i] for i in range(len(methods))},
        "ground_truth": ground_truth,
        "projections": projections,
        "pruning_intervals": {
            "Adam": pruning_interval,
            "ADMM": pruning_interval_admm
        },
        "iterations": {
            "Adam": adam_iterations,
            "ADMM": admm_iterations
        }
    }
    
    # Print timing and metrics comparison
    print("\n===== Optimization Comparison =====")
    
    # Timing comparison
    print("\nTiming Comparison:")
    for method, timing in results["times"].items():
        print(f"  {method}: {timing:.2f} seconds")
    
    # Points comparison
    print("\nFinal Number of Points:")
    for method, num_points in results["points_final"].items():
        print(f"  {method}: {num_points} points")
    
    # Metrics comparison
    print("\nMetrics Comparison:")
    print(f"{'Metric':<20} {'Adam':<12} {'ADMM':<12} {'Diff (ADMM-Adam)':<12} {'% Change':<12}")
    print("-" * 68)
    
    # First print standard metrics
    standard_metrics = ['iou', 'dice', 'precision', 'recall', 'f1', 'mse', 'psnr']
    special_metrics = ['mse_nonzero', 'psnr_nonzero']
    
    # Sort metrics to show standard ones first, then special ones with emphasis
    for metric in standard_metrics:
        if metric in adam_metrics:
            adam_val = adam_metrics[metric]
            admm_val = admm_metrics[metric]
            diff = admm_val - adam_val
            percent = (diff / adam_val) * 100 if adam_val != 0 else float('inf')
            print(f"{metric:<20} {adam_val:<12.6f} {admm_val:<12.6f} {diff:<12.6f} {percent:+<12.2f}%")
    
    # Print a separator
    print("-" * 68)
    
    # Then print special non-zero metrics with emphasis
    for metric in special_metrics:
        if metric in adam_metrics:
            adam_val = adam_metrics[metric]
            admm_val = admm_metrics[metric]
            diff = admm_val - adam_val
            percent = (diff / adam_val) * 100 if adam_val != 0 else float('inf')
            print(f"{metric:<20} {adam_val:<12.6f} {admm_val:<12.6f} {diff:<12.6f} {percent:+<12.2f}%")
    
    # Generate plots if requested
    if make_plots:
        generate_comparison_plots(results)
    
    return results

def generate_comparison_plots(results):
    """
    Generate plots comparing the Adam and ADMM optimization methods.
    
    Args:
        results (dict): Dictionary containing comparison results
    """
    # Extract data
    adam_volume = results["volumes"]["Adam"]
    admm_volume = results["volumes"]["ADMM"]
    ground_truth = results["ground_truth"]
    
    # 1. Side-by-side volume comparison
    print("Generating side-by-side volume comparison...")
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Maximum intensity projections
    gt_mip = np.max(ground_truth.cpu().numpy(), axis=0)
    adam_mip = np.max(adam_volume.cpu().numpy(), axis=0)
    admm_mip = np.max(admm_volume.cpu().numpy(), axis=0)
    
    im0 = axes[0].imshow(gt_mip, cmap='viridis')
    axes[0].set_title("Ground Truth")
    plt.colorbar(im0, ax=axes[0])
    
    im1 = axes[1].imshow(adam_mip, cmap='viridis')
    axes[1].set_title(f"Adam ({results['points_final']['Adam']} points)")
    plt.colorbar(im1, ax=axes[1])
    
    im2 = axes[2].imshow(admm_mip, cmap='viridis')
    axes[2].set_title(f"ADMM ({results['points_final']['ADMM']} points)")
    plt.colorbar(im2, ax=axes[2])
    
    plt.tight_layout()
    plt.savefig("img/comparison_volumes_mip.png")
    plt.close()
    
    # 2. Loss history comparison
    print("Generating loss history comparison...")
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Normalize iterations to percentage for fair comparison
    adam_iterations = len(results["loss_history"]["Adam"])
    admm_iterations = len(results["loss_history"]["ADMM"])
    
    adam_x = np.linspace(0, 100, adam_iterations)
    admm_x = np.linspace(0, 100, admm_iterations)
    
    ax.plot(adam_x, results["loss_history"]["Adam"], label=f"Adam ({results['iterations']['Adam']} iterations)")
    ax.plot(admm_x, results["loss_history"]["ADMM"], label=f"ADMM ({results['iterations']['ADMM']} iterations)")
    
    ax.set_xlabel("Progress (%)")
    ax.set_ylabel("Loss")
    ax.set_yscale('log')
    ax.set_title("Loss Comparison (Normalized to Iteration Percentage)")
    ax.legend()
    ax.grid(True)
    
    plt.tight_layout()
    plt.savefig("img/comparison_loss.png")
    plt.close()
    
    # 3. Points history comparison
    print("Generating points history comparison...")
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Normalize iterations to percentage for fair comparison
    adam_pruning_intervals = np.array([i * results["pruning_intervals"]["Adam"] for i in range(len(results["points_history"]["Adam"]))])
    admm_pruning_intervals = np.array([i * results["pruning_intervals"]["ADMM"] for i in range(len(results["points_history"]["ADMM"]))])
    
    adam_x = 100 * adam_pruning_intervals / results["iterations"]["Adam"]
    admm_x = 100 * admm_pruning_intervals / results["iterations"]["ADMM"]
    
    ax.plot(adam_x, results["points_history"]["Adam"], 'o-', label="Adam")
    ax.plot(admm_x, results["points_history"]["ADMM"], 's-', label="ADMM")
    
    ax.set_xlabel("Progress (%)")
    ax.set_ylabel("Number of Points")
    ax.set_title("Points Pruning Comparison")
    ax.legend()
    ax.grid(True)
    
    plt.tight_layout()
    plt.savefig("img/comparison_points.png")
    plt.close()
    
    # 4. Metrics comparison bar chart
    print("Generating metrics comparison chart...")
    metrics = results["metrics"]
    
    # Create bar chart for standard metrics
    standard_metrics = ['iou', 'dice', 'precision', 'recall', 'f1', 'mse', 'psnr']
    available_metrics = [m for m in standard_metrics if m in metrics["Adam"]]
    
    x = np.arange(len(available_metrics))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    adam_values = [metrics["Adam"][m] for m in available_metrics]
    admm_values = [metrics["ADMM"][m] for m in available_metrics]
    
    rects1 = ax.bar(x - width/2, adam_values, width, label="Adam")
    rects2 = ax.bar(x + width/2, admm_values, width, label="ADMM")
    
    ax.set_ylabel("Value")
    ax.set_title("Standard Metrics Comparison")
    ax.set_xticks(x)
    ax.set_xticklabels(available_metrics)
    ax.legend()
    
    # Add values on top of bars
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f"{height:.3f}",
                        xy=(rect.get_x() + rect.get_width()/2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom',
                        fontsize=8)
    
    autolabel(rects1)
    autolabel(rects2)
    
    plt.tight_layout()
    plt.savefig("img/comparison_metrics_standard.png")
    plt.close()
    
    # Create separate chart for non-zero metrics
    special_metrics = ['mse_nonzero', 'psnr_nonzero']
    available_special = [m for m in special_metrics if m in metrics["Adam"]]
    
    if available_special:
        x = np.arange(len(available_special))
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        adam_values = [metrics["Adam"][m] for m in available_special]
        admm_values = [metrics["ADMM"][m] for m in available_special]
        
        rects1 = ax.bar(x - width/2, adam_values, width, label="Adam", color='darkblue')
        rects2 = ax.bar(x + width/2, admm_values, width, label="ADMM", color='darkred')
        
        ax.set_ylabel("Value")
        ax.set_title("Non-Zero Region Metrics Comparison\n(Only considering non-zero pixels in target image)")
        ax.set_xticks(x)
        
        # Make the labels more descriptive
        better_labels = {
            'mse_nonzero': 'MSE (non-zero only)', 
            'psnr_nonzero': 'PSNR (non-zero only)'
        }
        ax.set_xticklabels([better_labels.get(m, m) for m in available_special])
        ax.legend()
        
        autolabel(rects1)
        autolabel(rects2)
        
        plt.tight_layout()
        plt.savefig("img/comparison_metrics_nonzero.png")
        plt.close()
        
        # Special chart for PSNR comparison
        if 'psnr' in metrics["Adam"] and 'psnr_nonzero' in metrics["Adam"]:
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Group data differently - group by method
            metrics_to_show = ['psnr', 'psnr_nonzero']
            x = np.arange(len(metrics_to_show))
            
            # Get values
            adam_psnr = [metrics["Adam"][m] for m in metrics_to_show]
            admm_psnr = [metrics["ADMM"][m] for m in metrics_to_show]
            
            # Plot
            width = 0.35
            rects1 = ax.bar(x - width/2, adam_psnr, width, label="Adam")
            rects2 = ax.bar(x + width/2, admm_psnr, width, label="ADMM")
            
            ax.set_ylabel("PSNR (dB)")
            ax.set_title("PSNR Comparison: All Pixels vs. Non-Zero Pixels Only")
            ax.set_xticks(x)
            ax.set_xticklabels(['Regular PSNR', 'PSNR (non-zero only)'])
            ax.legend()
            
            autolabel(rects1)
            autolabel(rects2)
            
            plt.tight_layout()
            plt.savefig("img/comparison_psnr_special.png")
            plt.close()
    
    # 5. Timing comparison
    print("Generating timing comparison...")
    fig, ax = plt.subplots(figsize=(8, 6))
    
    methods = list(results["times"].keys())
    times = [results["times"][m] for m in methods]
    
    ax.bar(methods, times)
    ax.set_ylabel("Time (seconds)")
    ax.set_title("Execution Time Comparison")
    
    # Add values on top of bars
    for i, v in enumerate(times):
        ax.text(i, v + 0.1, f"{v:.2f}s", ha='center')
    
    plt.tight_layout()
    plt.savefig("img/comparison_timing.png")
    plt.close()
    
    print("All comparison plots generated successfully!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Compare Adam and ADMM optimization methods for LArTPC reconstruction')
    parser.add_argument('--shape', type=int, default=100, help='Size of the cubic volume (shape x shape x shape)')
    parser.add_argument('--planes', type=int, default=3, help='Number of projection planes')
    parser.add_argument('--adam-iters', type=int, default=1000, help='Number of iterations for Adam optimizer')
    parser.add_argument('--admm-iters', type=int, default=200, help='Number of iterations for ADMM optimizer')
    parser.add_argument('--noise', type=float, default=0.0, help='Level of noise to add to projections')
    parser.add_argument('--no-plots', action='store_true', help='Disable plot generation')
    
    args = parser.parse_args()
    
    volume_shape = (args.shape, args.shape, args.shape)
    
    compare_optimization_methods(
        volume_shape=volume_shape,
        num_planes=args.planes,
        adam_iterations=args.adam_iters,
        admm_iterations=args.admm_iters,
        noise_level=args.noise,
        make_plots=not args.no_plots
    ) 