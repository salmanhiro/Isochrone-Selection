# Isochrone-Selection

A Python library for selecting stars around isochrones in Color-Magnitude Diagrams (CMDs). This tool addresses the non-trivial task of identifying stellar cluster members based on their proximity to theoretical isochrone curves.

## Overview

In stellar astronomy, isochrones are theoretical curves in a Color-Magnitude Diagram that represent stars of the same age but different masses. Selecting observed stars that lie near an isochrone is a fundamental task in stellar population studies, but it's far from trivial due to:

- Observational uncertainties and scatter
- Field star contamination
- Complex isochrone shapes (e.g., main sequence turn-off, red giant branch)
- Choice of appropriate distance metrics

This library provides efficient tools to perform isochrone-based star selection using various methods.

## Features

- **Distance-based selection**: Select stars within a specified perpendicular or Euclidean distance from an isochrone
- **Range-based selection**: Select stars within rectangular color-magnitude windows around the isochrone
- **Efficient algorithms**: Uses KD-trees for fast nearest-neighbor searches
- **Flexible metrics**: Support for multiple distance calculation methods
- **Well-tested**: Comprehensive unit tests ensure reliability

## Installation

```bash
pip install -r requirements.txt
```

### Requirements

- Python >= 3.7
- NumPy >= 1.20.0
- SciPy >= 1.7.0
- Matplotlib >= 3.3.0 (for visualization)

## Quick Start

```python
import numpy as np
from isochrone_selection import select_stars

# Define your isochrone (color and magnitude)
iso_color = np.array([0.0, 0.5, 1.0, 1.5, 2.0])
iso_mag = np.array([10.0, 11.0, 12.0, 13.0, 14.0])

# Define your observed stars
star_color = np.array([0.3, 0.7, 1.2, 3.0])
star_mag = np.array([10.5, 11.3, 12.1, 15.0])

# Select stars within 0.3 mag of the isochrone
selected_mask, distances = select_stars(
    star_color, star_mag,
    iso_color, iso_mag,
    threshold=0.3,
    metric='perpendicular'
)

print(f"Selected {np.sum(selected_mask)} out of {len(star_color)} stars")
```

## Usage

### Method 1: Distance-based Selection

The most flexible method, selecting stars based on their distance to the isochrone:

```python
from isochrone_selection import select_stars

# Select stars within 0.2 mag of the isochrone
mask, distances = select_stars(
    star_color, star_mag,
    isochrone_color, isochrone_mag,
    threshold=0.2,
    metric='perpendicular'  # or 'euclidean'
)

selected_stars_color = star_color[mask]
selected_stars_mag = star_mag[mask]
```

### Method 2: Color-Magnitude Range Selection

Select stars within a rectangular window around the isochrone:

```python
from isochrone_selection import select_stars_color_range

# Select stars within ±0.1 in color and ±0.25 in magnitude
mask = select_stars_color_range(
    star_color, star_mag,
    isochrone_color, isochrone_mag,
    color_threshold=0.1,
    mag_threshold=0.25
)
```

## Running the Example

The repository includes a complete example that demonstrates the functionality:

```bash
python example.py
```

This will:
1. Generate a synthetic isochrone (main sequence + red giant branch)
2. Create synthetic stellar data (cluster members + field stars)
3. Apply different selection methods
4. Generate visualization plots showing the results
5. Calculate completeness and contamination metrics

The example produces four plots:
- `cmd_original.png`: The original CMD with all stars
- `cmd_distance_selection.png`: Stars selected by distance-based method
- `cmd_range_selection.png`: Stars selected by range-based method
- `distance_distribution.png`: Histogram of distances from the isochrone

## Running Tests

The library includes comprehensive unit tests:

```bash
python -m unittest test_isochrone_selection.py -v
```

## API Reference

### `select_stars(star_color, star_mag, isochrone_color, isochrone_mag, threshold, metric='perpendicular')`

Select stars within a specified distance from an isochrone.

**Parameters:**
- `star_color` (array-like): Color values of observed stars
- `star_mag` (array-like): Magnitude values of observed stars
- `isochrone_color` (array-like): Color values of the isochrone
- `isochrone_mag` (array-like): Magnitude values of the isochrone
- `threshold` (float): Maximum distance for selection
- `metric` (str): Distance metric ('perpendicular' or 'euclidean')

**Returns:**
- `mask` (ndarray of bool): Selection mask (True = selected)
- `distances` (ndarray): Distance of each star from the isochrone

### `select_stars_color_range(star_color, star_mag, isochrone_color, isochrone_mag, color_threshold, mag_threshold)`

Select stars within a rectangular region around the isochrone.

**Parameters:**
- `star_color`, `star_mag`: Observed star positions
- `isochrone_color`, `isochrone_mag`: Isochrone curve
- `color_threshold` (float): Maximum color deviation (±)
- `mag_threshold` (float): Maximum magnitude deviation (±)

**Returns:**
- `mask` (ndarray of bool): Selection mask

### `perpendicular_distance(points, isochrone_color, isochrone_mag)`

Calculate perpendicular distances from points to an isochrone curve.

### `euclidean_distance(points, isochrone_color, isochrone_mag)`

Calculate Euclidean distances to the nearest point on an isochrone.

## Use Cases

This library is useful for:

- **Stellar cluster studies**: Identifying cluster members in open or globular clusters
- **Isochrone fitting**: Cleaning data before fitting theoretical isochrones
- **Age determination**: Selecting stars for age estimation from CMD fitting
- **Stellar population analysis**: Isolating specific stellar populations
- **Quality control**: Filtering photometric data based on expected stellar loci

## Algorithm Details

The library uses scipy's cKDTree for efficient nearest-neighbor searches, making it suitable for large datasets (10⁴-10⁶ stars). The perpendicular distance is approximated by finding the nearest point on the isochrone, which works well for smooth, well-sampled isochrones.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Citation

If you use this library in your research, please cite it appropriately.

## Acknowledgments

This tool was developed to address the practical challenges in stellar population studies where selecting stars around isochrones is a fundamental but non-trivial task.
