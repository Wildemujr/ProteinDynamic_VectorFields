"""
This script parses NMD files containing protein structure information, computes spatial derivatives, 
and calculates divergence and curl for 2D vector fields representing protein motions.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import KDTree


def parse_nmd(filename):
    """
    Parse an NMD file and extract relevant data.
    
    Parameters:
        filename (str): The path to the NMD file to be parsed.

    Returns:
        dict: A dictionary containing parsed data such as mode, scale_factor, displacements, and coordinates.

    Raises:
        Exception: If the NMD file does not contain a "mode" line.
    """
    with open(filename) as file:
        data = {}
        for line in file:
            words = line.split()
            if words[0] == 'mode':
                data['mode'] = int(words[1])
                data['scale_factor'] = float(words[2])
                data['displacements'] = np.array(words[3:], dtype=float).reshape(-1, 3)
            elif words[0] == 'coordinates':
                data['coordinates'] = np.array(words[1:], dtype=float).reshape(-1, 3)
            else:
                data[words[0]] = words[1:]

    if 'mode' not in data:
        raise Exception(f"File {filename} does not contain a 'mode' line.")
    
    return data


def compute_derivatives(coordinates, displacements):
    """
    Compute spatial derivatives using nearest neighbors for the given coordinates and displacements.
    
    Parameters:
        coordinates (np.ndarray): An array of coordinates representing points in the vector field.
        displacements (np.ndarray): An array of displacement vectors corresponding to the coordinates.

    Returns:
        tuple: A tuple containing arrays representing the spatial derivatives dP_dx, dP_dy, dQ_dx, dQ_dy.
    """
    tree = KDTree(coordinates)
    
    dP_dx = np.zeros(coordinates.shape[0])
    dP_dy = np.zeros(coordinates.shape[0])
    dQ_dx = np.zeros(coordinates.shape[0])
    dQ_dy = np.zeros(coordinates.shape[0])
    
    for i, coord in enumerate(coordinates):
        _, indices = tree.query(coord, 2)  # Get two nearest points (includes the point itself)
        neighbor = coordinates[indices[1]]
        
        dx = neighbor[0] - coord[0]
        dy = neighbor[1] - coord[1]
        
        dP = displacements[indices[1], 0] - displacements[i, 0]
        dQ = displacements[indices[1], 1] - displacements[i, 1]
        
        dP_dx[i] = dP / dx if dx != 0 else 0
        dP_dy[i] = dP / dy if dy != 0 else 0
        dQ_dx[i] = dQ / dx if dx != 0 else 0
        dQ_dy[i] = dQ / dy if dy != 0 else 0
    
    return dP_dx, dP_dy, dQ_dx, dQ_dy


def compute_divergence_curl(coordinates, displacements):
    """
    Compute the divergence and curl of a 2D vector field based on spatial derivatives.

    Parameters:
        coordinates (np.ndarray): An array of coordinates representing points in the vector field.
        displacements (np.ndarray): An array of displacement vectors corresponding to the coordinates.

    Returns:
        tuple: A tuple containing arrays representing the divergence and curl of the vector field.
    """
    dP_dx, dP_dy, dQ_dx, dQ_dy = compute_derivatives(coordinates, displacements)
    divergence = dP_dx + dQ_dy
    curl = dQ_dx - dP_dy
    return divergence, curl


def visualize_divergence_curl_enhanced(coordinates, divergence, curl, ax, title):
    """
    Visualize the divergence and curl of a vector field with enhanced visual features.

    Parameters:
        coordinates (np.ndarray): An array of coordinates representing points in the vector field.
        divergence (np.ndarray): An array representing the divergence of the vector field.
        curl (np.ndarray): An array representing the curl of the vector field.
        ax (matplotlib.axes.Axes): The axes on which to plot the visualization.
        title (str): The title of the plot.
    """
    scatter1 = ax.scatter(coordinates[:, 0], coordinates[:, 1], c=divergence, s=15, cmap="RdBu_r", 
                          vmin=-np.percentile(np.abs(divergence), 95), vmax=np.percentile(np.abs(divergence), 95), 
                          edgecolors='k', linewidth=0.5, label="Divergence")
    scatter2 = ax.scatter(coordinates[:, 0], coordinates[:, 1], c=curl, s=15, cmap="RdBu_r", 
                          vmin=-np.percentile(np.abs(curl), 95), vmax=np.percentile(np.abs(curl), 95), 
                          linewidth=0.5, label="Curl", marker='o')
    ax.set_title(title)
    ax.set_xlabel("Coordinate 1")
    ax.set_ylabel("Coordinate 2")
    ax.grid(True)
    return scatter1, scatter2




def main():
    # Load the provided NMD file and parse the data
    if len(sys.argv) < 2:
        print("Usage: python script.py <pdb_id>")
        sys.exit(1)

    # Load the provided NMD file and parse the data
    pdb_id = sys.argv[1]
    cwd = os.getcwd() 
    full_path = os.path.join(cwd, f"ANM_Modes/{pdb_id.upper()}_ANM_Modes_1_through_All.nmd")
    parsed_data = parse_nmd(full_path)

    # Set the scale factor for the displacements
    scale_factor = 80.0

    # Extracting the 2D coordinates and displacement vectors
    coordinates_2d = parsed_data['coordinates'][:, :2]
    displacements_2d = parsed_data['displacements'][:, :2]

    # Extracting the 2D projections for XZ and YZ planes
    coordinates_xz = parsed_data['coordinates'][:, [0, 2]]
    displacements_xz = parsed_data['displacements'][:, [0, 2]]

    coordinates_yz = parsed_data['coordinates'][:, [1, 2]]
    displacements_yz = parsed_data['displacements'][:, [1, 2]]

    # Visualizing all three projections
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # XY-plane
    axes[0].quiver(coordinates_2d[:, 0], coordinates_2d[:, 1], 
                   scale_factor * displacements_2d[:, 0], scale_factor * displacements_2d[:, 1], 
                   angles='xy', scale_units='xy', scale=1, color='blue', width=0.003)
    axes[0].scatter(coordinates_2d[:, 0], coordinates_2d[:, 1], color='red', s=15)
    axes[0].set_title("XY Plane Projection")
    axes[0].set_xlabel("X Coordinate")
    axes[0].set_ylabel("Y Coordinate")
    axes[0].grid(True)

    # XZ-plane
    axes[1].quiver(coordinates_xz[:, 0], coordinates_xz[:, 1], 
                   scale_factor * displacements_xz[:, 0], scale_factor * displacements_xz[:, 1], 
                   angles='xy', scale_units='xy', scale=1, color='blue', width=0.003)
    axes[1].scatter(coordinates_xz[:, 0], coordinates_xz[:, 1], color='red', s=15)
    axes[1].set_title("XZ Plane Projection")
    axes[1].set_xlabel("X Coordinate")
    axes[1].set_ylabel("Z Coordinate")
    axes[1].grid(True)

    # YZ-plane
    axes[2].quiver(coordinates_yz[:, 0], coordinates_yz[:, 1], 
                   scale_factor * displacements_yz[:, 0], scale_factor * displacements_yz[:, 1], 
                   angles='xy', scale_units='xy', scale=1, color='blue', width=0.003)
    axes[2].scatter(coordinates_yz[:, 0], coordinates_yz[:, 1], color='red', s=15)
    axes[2].set_title("YZ Plane Projection")
    axes[2].set_xlabel("Y Coordinate")
    axes[2].set_ylabel("Z Coordinate")
    axes[2].grid(True)

    plt.tight_layout()
    plt.show()


    # Computing the Divergence and Curl for each projection
    divergence_xy, curl_xy = compute_divergence_curl(coordinates_2d, displacements_2d)
    divergence_xz, curl_xz = compute_divergence_curl(coordinates_xz, displacements_xz)
    divergence_yz, curl_yz = compute_divergence_curl(coordinates_yz, displacements_yz)

    # Setting up the figure
    fig, axes = plt.subplots(3, 2, figsize=(18, 15))

    # XY-plane
    visualize_divergence_curl_enhanced(coordinates_2d, divergence_xy, curl_xy, axes[0, 0], "XY Plane Divergence")
    visualize_divergence_curl_enhanced(coordinates_2d, curl_xy, divergence_xy, axes[0, 1], "XY Plane Curl")

    # XZ-plane
    visualize_divergence_curl_enhanced(coordinates_xz, divergence_xz, curl_xz, axes[1, 0], "XZ Plane Divergence")
    visualize_divergence_curl_enhanced(coordinates_xz, curl_xz, divergence_xz, axes[1, 1], "XZ Plane Curl")

    # YZ-plane
    visualize_divergence_curl_enhanced(coordinates_yz, divergence_yz, curl_yz, axes[2, 0], "YZ Plane Divergence")
    visualize_divergence_curl_enhanced(coordinates_yz, curl_yz, divergence_yz, axes[2, 1], "YZ Plane Curl")

    # Add colorbars outside of the plots
    cbar_ax1 = fig.add_axes([0.14, 0.08, 0.35, 0.01])
    cbar_ax2 = fig.add_axes([0.58, 0.08, 0.35, 0.01])
    fig.colorbar(plt.cm.ScalarMappable(cmap="RdBu_r"), cax=cbar_ax1, orientation='horizontal', label="Divergence Value")
    fig.colorbar(plt.cm.ScalarMappable(cmap="RdBu_r"), cax=cbar_ax2, orientation='horizontal', label="Curl Value")

    # Adjust subplot spacing
    plt.subplots_adjust(wspace=0.3, hspace=0.3)
    plt.show()



if __name__ == "__main__":
    main()