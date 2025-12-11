"""
Example script demonstrating isochrone selection functionality.

This script creates a synthetic Color-Magnitude Diagram (CMD) with:
1. A theoretical isochrone
2. Synthetic stellar data around the isochrone
3. Background/field stars

It then demonstrates different selection methods and visualizes the results.
"""

import numpy as np
import matplotlib.pyplot as plt
from isochrone_selection import select_stars, select_stars_color_range


# Constants for synthetic data generation
COLOR_SCATTER_STD = 0.05  # Standard deviation of color scatter around isochrone
MAG_SCATTER_STD = 0.15    # Standard deviation of magnitude scatter around isochrone
MAG_RANGE_EXTENSION = 2.0  # Magnitude range extension for field stars


def generate_synthetic_isochrone():
    """Generate a synthetic isochrone curve."""
    # Create a realistic-looking isochrone curve (e.g., main sequence + red giant branch)
    color = np.concatenate([
        np.linspace(0.0, 0.6, 30),  # Blue part of main sequence
        np.linspace(0.6, 1.5, 20)    # Red giant branch
    ])
    
    # Main sequence: magnitude increases (gets fainter) with color
    ms_mag = 4 + 8 * color[:30]**1.5
    
    # Red giant branch: magnitude decreases (gets brighter) with color
    rgb_color = color[30:] - 0.6
    rgb_mag = 10 + 2 * np.sin(rgb_color * 3) - rgb_color * 2
    
    magnitude = np.concatenate([ms_mag, rgb_mag])
    
    return color, magnitude


def generate_synthetic_stars(isochrone_color, isochrone_mag, n_cluster=500, n_field=1000):
    """
    Generate synthetic star data.
    
    Parameters
    ----------
    isochrone_color : array-like
        Isochrone color values
    isochrone_mag : array-like
        Isochrone magnitude values
    n_cluster : int
        Number of cluster stars (near the isochrone)
    n_field : int
        Number of field stars (random background)
    
    Returns
    -------
    star_color : ndarray
        Color values of all stars
    star_mag : ndarray
        Magnitude values of all stars
    true_members : ndarray of bool
        True membership (for validation)
    """
    # Generate cluster stars scattered around the isochrone
    # Randomly sample points along the isochrone
    indices = np.random.choice(len(isochrone_color), n_cluster)
    cluster_color_base = isochrone_color[indices]
    cluster_mag_base = isochrone_mag[indices]
    
    # Add scatter around the isochrone
    color_scatter = np.random.normal(0, COLOR_SCATTER_STD, n_cluster)
    mag_scatter = np.random.normal(0, MAG_SCATTER_STD, n_cluster)
    
    cluster_color = cluster_color_base + color_scatter
    cluster_mag = cluster_mag_base + mag_scatter
    
    # Generate field stars uniformly distributed
    color_min, color_max = isochrone_color.min() - 0.5, isochrone_color.max() + 0.5
    mag_min, mag_max = float(isochrone_mag.min() - MAG_RANGE_EXTENSION), float(isochrone_mag.max() + MAG_RANGE_EXTENSION)
    
    # Ensure valid range
    if mag_min > mag_max:
        mag_min, mag_max = mag_max, mag_min
    
    field_color = np.random.uniform(color_min, color_max, n_field)
    field_mag = np.random.uniform(mag_min, mag_max, n_field)
    
    # Combine all stars
    star_color = np.concatenate([cluster_color, field_color])
    star_mag = np.concatenate([cluster_mag, field_mag])
    
    # Track true membership
    true_members = np.concatenate([
        np.ones(n_cluster, dtype=bool),
        np.zeros(n_field, dtype=bool)
    ])
    
    return star_color, star_mag, true_members


def plot_cmd(star_color, star_mag, isochrone_color, isochrone_mag, 
             selected_mask=None, title="Color-Magnitude Diagram"):
    """Plot a Color-Magnitude Diagram."""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot all stars
    if selected_mask is not None:
        # Plot unselected stars in gray
        ax.scatter(star_color[~selected_mask], star_mag[~selected_mask], 
                  c='lightgray', s=10, alpha=0.5, label='Unselected stars')
        # Plot selected stars in blue
        ax.scatter(star_color[selected_mask], star_mag[selected_mask], 
                  c='blue', s=20, alpha=0.7, label='Selected stars')
    else:
        ax.scatter(star_color, star_mag, c='blue', s=10, alpha=0.5, label='Stars')
    
    # Plot isochrone
    ax.plot(isochrone_color, isochrone_mag, 'r-', linewidth=2, label='Isochrone')
    
    # Formatting
    ax.set_xlabel('Color (e.g., B-V)', fontsize=12)
    ax.set_ylabel('Magnitude', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.invert_yaxis()  # Brighter stars (lower mag) at top
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    return fig, ax


def main():
    """Main example demonstrating isochrone selection."""
    print("=" * 60)
    print("Isochrone Selection Example")
    print("=" * 60)
    
    # Generate synthetic data
    print("\n1. Generating synthetic isochrone...")
    iso_color, iso_mag = generate_synthetic_isochrone()
    print(f"   Created isochrone with {len(iso_color)} points")
    
    print("\n2. Generating synthetic stellar data...")
    star_color, star_mag, true_members = generate_synthetic_stars(
        iso_color, iso_mag, n_cluster=500, n_field=1000
    )
    print(f"   Created {len(star_color)} stars")
    print(f"   - Cluster members: {np.sum(true_members)}")
    print(f"   - Field stars: {np.sum(~true_members)}")
    
    # Method 1: Distance-based selection
    print("\n3. Applying distance-based selection...")
    threshold = 0.3
    selected_distance, distances = select_stars(
        star_color, star_mag, iso_color, iso_mag, 
        threshold=threshold
    )
    n_selected = np.sum(selected_distance)
    print(f"   Threshold: {threshold}")
    print(f"   Selected stars: {n_selected}")
    
    # Calculate completeness and contamination
    true_positives = np.sum(selected_distance & true_members)
    false_positives = np.sum(selected_distance & ~true_members)
    false_negatives = np.sum(~selected_distance & true_members)
    
    completeness = true_positives / np.sum(true_members) * 100
    contamination = false_positives / n_selected * 100 if n_selected > 0 else 0
    
    print(f"   Completeness: {completeness:.1f}%")
    print(f"   Contamination: {contamination:.1f}%")
    
    # Method 2: Color-magnitude range selection
    print("\n4. Applying color-magnitude range selection...")
    color_threshold = 0.1
    mag_threshold = 0.25
    selected_range = select_stars_color_range(
        star_color, star_mag, iso_color, iso_mag,
        color_threshold=color_threshold, mag_threshold=mag_threshold
    )
    n_selected_range = np.sum(selected_range)
    print(f"   Color threshold: ±{color_threshold}")
    print(f"   Magnitude threshold: ±{mag_threshold}")
    print(f"   Selected stars: {n_selected_range}")
    
    # Calculate metrics for range selection
    true_positives_range = np.sum(selected_range & true_members)
    false_positives_range = np.sum(selected_range & ~true_members)
    
    completeness_range = true_positives_range / np.sum(true_members) * 100
    contamination_range = false_positives_range / n_selected_range * 100 if n_selected_range > 0 else 0
    
    print(f"   Completeness: {completeness_range:.1f}%")
    print(f"   Contamination: {contamination_range:.1f}%")
    
    # Create visualizations
    print("\n5. Creating visualizations...")
    
    # Plot 1: Original CMD
    fig1, ax1 = plot_cmd(star_color, star_mag, iso_color, iso_mag,
                         title="Original Color-Magnitude Diagram")
    plt.savefig('cmd_original.png', dpi=150, bbox_inches='tight')
    print("   Saved: cmd_original.png")
    
    # Plot 2: Distance-based selection
    fig2, ax2 = plot_cmd(star_color, star_mag, iso_color, iso_mag,
                         selected_mask=selected_distance,
                         title=f"Distance-based Selection (threshold={threshold})")
    plt.savefig('cmd_distance_selection.png', dpi=150, bbox_inches='tight')
    print("   Saved: cmd_distance_selection.png")
    
    # Plot 3: Range-based selection
    fig3, ax3 = plot_cmd(star_color, star_mag, iso_color, iso_mag,
                         selected_mask=selected_range,
                         title=f"Range-based Selection (Δcolor=±{color_threshold}, Δmag=±{mag_threshold})")
    plt.savefig('cmd_range_selection.png', dpi=150, bbox_inches='tight')
    print("   Saved: cmd_range_selection.png")
    
    # Plot 4: Distance distribution
    fig4, ax4 = plt.subplots(figsize=(10, 6))
    ax4.hist(distances[true_members], bins=50, alpha=0.7, label='True members', color='blue')
    ax4.hist(distances[~true_members], bins=50, alpha=0.7, label='Field stars', color='red')
    ax4.axvline(threshold, color='black', linestyle='--', linewidth=2, label=f'Threshold={threshold}')
    ax4.set_xlabel('Distance from Isochrone', fontsize=12)
    ax4.set_ylabel('Number of Stars', fontsize=12)
    ax4.set_title('Distribution of Distances from Isochrone', fontsize=14)
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    plt.savefig('distance_distribution.png', dpi=150, bbox_inches='tight')
    print("   Saved: distance_distribution.png")
    
    print("\n" + "=" * 60)
    print("Example completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
