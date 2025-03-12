#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Visualization of wire plane orientations and their intersections in a LArTPC detector.
This script helps understand what the plane angle input actually means in the context
of LArTPC reconstruction.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.colors as mcolors
import argparse

# Add the parent directory to the path to import the lartpc_reconstruction package
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lartpc_reconstruction import LArTPCReconstructor, Line3D


def visualize_wire_planes(
    plane_angles=None, volume_shape=(100, 100, 100), num_wires=10, save_path=None
):
    """
    Visualize wire plane orientations and their intersections in 3D.

    Args:
        plane_angles (dict, optional): Dictionary mapping plane_id to angle in radians.
            If None, uses default angles (0°, 90°, 60°).
        volume_shape (tuple): Shape of the detector volume
        num_wires (int): Number of wires to draw per plane
        save_path (str, optional): Path to save the figure. If None, displays the figure.
    """
    # Default plane angles if not provided
    if plane_angles is None:
        plane_angles = {
            0: 0.0,  # 0 degrees (vertical)
            1: np.pi / 2,  # 90 degrees (horizontal)
            2: np.pi / 3,  # 60 degrees
        }

    # Create a figure
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection="3d")

    # Set volume limits
    ax.set_xlim(0, volume_shape[0])
    ax.set_ylim(0, volume_shape[1])
    ax.set_zlim(0, volume_shape[2])

    # Define colors for each plane
    colors = {0: "red", 1: "green", 2: "blue", 3: "purple", 4: "orange", 5: "cyan"}

    # Define detector boundaries
    detector_corners = [
        [0, 0, 0],
        [0, 0, volume_shape[2]],
        [0, volume_shape[1], 0],
        [0, volume_shape[1], volume_shape[2]],
        [volume_shape[0], 0, 0],
        [volume_shape[0], 0, volume_shape[2]],
        [volume_shape[0], volume_shape[1], 0],
        [volume_shape[0], volume_shape[1], volume_shape[2]],
    ]

    # Draw detector boundaries (transparent box)
    for i in range(4):
        ax.plot(
            [detector_corners[i][0], detector_corners[i + 4][0]],
            [detector_corners[i][1], detector_corners[i + 4][1]],
            [detector_corners[i][2], detector_corners[i + 4][2]],
            "k-",
            alpha=0.3,
        )

    for i in range(0, 8, 4):
        ax.plot(
            [detector_corners[i][0], detector_corners[i + 1][0]],
            [detector_corners[i][1], detector_corners[i + 1][1]],
            [detector_corners[i][2], detector_corners[i + 1][2]],
            "k-",
            alpha=0.3,
        )
        ax.plot(
            [detector_corners[i][0], detector_corners[i + 2][0]],
            [detector_corners[i][1], detector_corners[i + 2][1]],
            [detector_corners[i][2], detector_corners[i + 2][2]],
            "k-",
            alpha=0.3,
        )
        ax.plot(
            [detector_corners[i + 3][0], detector_corners[i + 1][0]],
            [detector_corners[i + 3][1], detector_corners[i + 1][1]],
            [detector_corners[i + 3][2], detector_corners[i + 1][2]],
            "k-",
            alpha=0.3,
        )
        ax.plot(
            [detector_corners[i + 3][0], detector_corners[i + 2][0]],
            [detector_corners[i + 3][1], detector_corners[i + 2][1]],
            [detector_corners[i + 3][2], detector_corners[i + 2][2]],
            "k-",
            alpha=0.3,
        )

    # Draw coordinate axes
    origin = [0, 0, 0]
    ax.quiver(*origin, 20, 0, 0, color="red", arrow_length_ratio=0.1, label="X")
    ax.quiver(*origin, 0, 20, 0, color="green", arrow_length_ratio=0.1, label="Y")
    ax.quiver(*origin, 0, 0, 20, color="blue", arrow_length_ratio=0.1, label="Z")

    # Draw wire planes
    max_y = volume_shape[1]
    max_z = volume_shape[2]

    # Add an anode plane at x=0 (semi-transparent)
    y_grid, z_grid = np.meshgrid(np.linspace(0, max_y, 10), np.linspace(0, max_z, 10))
    x_grid = np.zeros_like(y_grid)
    ax.plot_surface(x_grid, y_grid, z_grid, color="gray", alpha=0.2)

    # Create the reconstructor to use its methods
    reconstructor = LArTPCReconstructor(
        volume_shape=volume_shape, plane_angles=plane_angles, debug=False
    )

    # Track line objects for later showing intersections
    all_lines = []

    for plane_id, theta in plane_angles.items():
        color = colors[plane_id % len(colors)]  # Use modulo to handle more than 6 planes

        # Calculate wire direction (perpendicular to the projection direction)
        # The projection direction is (0, -sin(theta), cos(theta))
        wire_direction = np.array([0.0, np.cos(theta), np.sin(theta)])

        # Display plane information
        angle_degrees = theta * 180 / np.pi
        ax.text(
            volume_shape[0] / 2,
            max_y * 1.1,
            max_z * (0.8 - 0.1 * plane_id),
            f"Plane {plane_id}: {angle_degrees:.1f}° ({theta:.2f} rad)",
            color=color,
            fontsize=12,
        )

        # Draw wire direction using a standard quiver instead of custom Arrow3D
        # This is more compatible with newer matplotlib versions
        ax.quiver(
            volume_shape[0] / 2, 
            max_y * 0.5, 
            max_z * 0.9 - 10 * plane_id,
            0,  # No x component
            20 * wire_direction[1],  # y component
            20 * wire_direction[2],  # z component
            color=color,
            arrow_length_ratio=0.1,
        )

        # Calculate wire positions
        if abs(np.sin(theta)) < 1e-6:  # Vertical wires (theta ≈ 0°)
            u_values = np.linspace(0, max_z, num_wires)
            for u in u_values:
                # Draw a vertical wire at position u
                ax.plot(
                    [0, 0],  # x coordinates (at anode)
                    [0, max_y],  # y coordinates (full span)
                    [u, u],  # z coordinates (constant)
                    color=color,
                    lw=1,
                    alpha=0.7,
                )
                
                # Create a Line3D object for later
                line = Line3D(
                    point=[0, 0, u],
                    direction=wire_direction/np.linalg.norm(wire_direction),
                    plane_id=plane_id,
                )
                all_lines.append(line)

        elif abs(np.cos(theta)) < 1e-6:  # Horizontal wires (theta ≈ 90°)
            u_values = np.linspace(0, max_y, num_wires)
            for u in u_values:
                # Draw a horizontal wire at position u
                ax.plot(
                    [0, 0],  # x coordinates (at anode)
                    [u, u],  # y coordinates (constant)
                    [0, max_z],  # z coordinates (full span)
                    color=color,
                    lw=1,
                    alpha=0.7,
                )
                
                # Create a Line3D object for later
                line = Line3D(
                    point=[0, u, 0],
                    direction=wire_direction/np.linalg.norm(wire_direction),
                    plane_id=plane_id,
                )
                all_lines.append(line)

        else:  # Angled wires
            # Project onto a rotated axis system
            u_min = reconstructor.solver.u_min_values[plane_id]
            u_max = max_y * np.cos(theta) + max_z * np.sin(theta)
            u_values = np.linspace(u_min, u_max, num_wires)

            # For each u value, find where the wire intersects the detector edges
            # We need to solve for (y,z) points where -sin(θ)y + cos(θ)z = u
            for u in u_values:
                # Find two points on this wire: one at y=0 and one at z=0
                # This gives us (0, z) and (y, 0) where:
                # cos(θ)z = u and -sin(θ)y = u
                if abs(np.cos(theta)) > 1e-6:
                    z_at_y0 = u / np.cos(theta)
                    if 0 <= z_at_y0 <= max_z:
                        point1 = [0, 0, z_at_y0]
                    else:
                        # If out of bounds, use y=max_y edge
                        z_at_maxy = (u + np.sin(theta) * max_y) / np.cos(theta)
                        if 0 <= z_at_maxy <= max_z:
                            point1 = [0, max_y, z_at_maxy]
                        else:
                            continue  # Wire doesn't intersect valid detector edges
                else:
                    point1 = None

                if abs(np.sin(theta)) > 1e-6:
                    y_at_z0 = -u / np.sin(theta)
                    if 0 <= y_at_z0 <= max_y:
                        point2 = [0, y_at_z0, 0]
                    else:
                        # If out of bounds, use z=max_z edge
                        y_at_maxz = (np.cos(theta) * max_z - u) / np.sin(theta)
                        if 0 <= y_at_maxz <= max_y:
                            point2 = [0, y_at_maxz, max_z]
                        else:
                            continue  # Wire doesn't intersect valid detector edges
                else:
                    point2 = None

                # If we have two valid intersection points, draw the wire
                if point1 and point2:
                    ax.plot(
                        [point1[0], point2[0]],
                        [point1[1], point2[1]],
                        [point1[2], point2[2]],
                        color=color,
                        lw=1,
                        alpha=0.7,
                    )

                    # Create a Line3D object for later
                    line = Line3D(
                        point=point1,
                        direction=wire_direction / np.linalg.norm(wire_direction),
                        plane_id=plane_id,
                    )
                    all_lines.append(line)

    # Now show some example intersection points between planes
    if len(plane_angles) >= 2:
        # Group lines by plane_id
        lines_by_plane = {}
        for line in all_lines:
            if line.plane_id not in lines_by_plane:
                lines_by_plane[line.plane_id] = []
            lines_by_plane[line.plane_id].append(line)

        # Find intersections between each pair of planes
        for plane_id1 in plane_angles.keys():
            for plane_id2 in plane_angles.keys():
                if plane_id1 < plane_id2 and plane_id1 in lines_by_plane and plane_id2 in lines_by_plane:
                    # Show intersections for a small subset of lines (to avoid clutter)
                    subset1 = lines_by_plane[plane_id1][:2]
                    subset2 = lines_by_plane[plane_id2][:2]

                    for line1 in subset1:
                        for line2 in subset2:
                            # Calculate intersection
                            p1, d1 = line1.point, line1.direction
                            p2, d2 = line2.point, line2.direction

                            # Cross product of directions
                            cross = np.cross(d1, d2)
                            cross_norm = np.linalg.norm(cross)

                            # If lines are not parallel
                            if cross_norm > 1e-10:
                                # Calculate parameters of closest points
                                p2_p1 = p2 - p1

                                # Solve the system of equations
                                # (p1 + t1*d1 - p2 - t2*d2) is parallel to cross
                                mat = np.column_stack([d1, -d2])
                                b = p2 - p1

                                # Solve for t1, t2 using least squares
                                t1, t2 = np.linalg.lstsq(mat, b, rcond=None)[0]

                                # Calculate closest points
                                closest1 = p1 + t1 * d1
                                closest2 = p2 + t2 * d2

                                # Calculate midpoint as intersection
                                intersection = (closest1 + closest2) / 2

                                # Only show if intersection is inside or near the detector
                                if (
                                    0 <= intersection[0] <= volume_shape[0]
                                    and -10 <= intersection[1] <= volume_shape[1] + 10
                                    and -10 <= intersection[2] <= volume_shape[2] + 10
                                ):
                                    # Blend the colors of the two planes
                                    color1 = mcolors.to_rgb(colors[plane_id1 % len(colors)])
                                    color2 = mcolors.to_rgb(colors[plane_id2 % len(colors)])
                                    mixed_color = np.array(color1) * 0.5 + np.array(color2) * 0.5

                                    # Draw the intersection point
                                    ax.scatter(
                                        intersection[0],
                                        intersection[1],
                                        intersection[2],
                                        color=mixed_color,
                                        s=50,
                                        alpha=0.7,
                                    )

    # Set labels and title
    ax.set_xlabel("X (Drift Direction)")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("Wire Plane Orientations and Intersections")

    # Add legend for the planes
    for plane_id in plane_angles.keys():
        color = colors[plane_id % len(colors)]
        angle_deg = plane_angles[plane_id] * 180 / np.pi
        ax.plot([], [], color=color, label=f"Plane {plane_id}: {angle_deg:.1f}°")
    ax.legend()

    # Set aspect ratio to be equal
    # This is a workaround as ax.set_aspect('equal') doesn't work well with 3D
    max_range = max([volume_shape[0], volume_shape[1], volume_shape[2]])
    mid_x = volume_shape[0] / 2
    mid_y = volume_shape[1] / 2
    mid_z = volume_shape[2] / 2
    ax.set_xlim(mid_x - max_range / 2, mid_x + max_range / 2)
    ax.set_ylim(mid_y - max_range / 2, mid_y + max_range / 2)
    ax.set_zlim(mid_z - max_range / 2, mid_z + max_range / 2)

    # Set the view angle
    ax.view_init(elev=30, azim=-60)

    # Save or display the figure
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Figure saved to {save_path}")
    else:
        plt.tight_layout()
        plt.show()

    return fig


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Visualize wire plane orientations and intersections"
    )
    parser.add_argument(
        "--size",
        type=int,
        default=100,
        help="Size of the cubic volume (size x size x size)",
    )
    parser.add_argument(
        "--wires", type=int, default=10, help="Number of wires to draw per plane"
    )
    parser.add_argument(
        "--save",
        type=str,
        default="wire_planes.png",
        help='Path to save the figure (set to "show" to display instead)',
    )
    parser.add_argument(
        "--angles",
        type=str,
        default="0,90,60",
        help="Comma-separated list of plane angles in degrees",
    )

    args = parser.parse_args()

    # Parse plane angles
    angles_list = [float(a) for a in args.angles.split(",")]
    plane_angles = {i: angle * np.pi / 180 for i, angle in enumerate(angles_list)}

    volume_shape = (args.size, args.size, args.size)
    save_path = None if args.save == "show" else args.save

    visualize_wire_planes(
        plane_angles=plane_angles,
        volume_shape=volume_shape,
        num_wires=args.wires,
        save_path=save_path,
    )