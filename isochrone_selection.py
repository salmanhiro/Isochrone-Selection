"""
Isochrone Selection Module

This module provides tools for selecting stars around an isochrone curve
in a Color-Magnitude Diagram (CMD).
"""

import numpy as np
from scipy.interpolate import interp1d
from scipy.spatial import cKDTree


def perpendicular_distance(points, isochrone_color, isochrone_mag):
    """
    Calculate the perpendicular distance from points to an isochrone curve.
    
    Parameters
    ----------
    points : array-like, shape (n, 2)
        Array of points with columns [color, magnitude]
    isochrone_color : array-like
        Color values of the isochrone (e.g., B-V, G-R)
    isochrone_mag : array-like
        Magnitude values of the isochrone
    
    Returns
    -------
    distances : ndarray
        Perpendicular distance of each point to the isochrone
    """
    points = np.asarray(points)
    isochrone_color = np.asarray(isochrone_color)
    isochrone_mag = np.asarray(isochrone_mag)
    
    # Ensure isochrone points are sorted by color
    sort_idx = np.argsort(isochrone_color)
    isochrone_color = isochrone_color[sort_idx]
    isochrone_mag = isochrone_mag[sort_idx]
    
    # Build KD-tree for efficient nearest neighbor search
    isochrone_points = np.column_stack([isochrone_color, isochrone_mag])
    tree = cKDTree(isochrone_points)
    
    # Find nearest points on isochrone
    distances, indices = tree.query(points)
    
    return distances


def euclidean_distance(points, isochrone_color, isochrone_mag):
    """
    Calculate the Euclidean distance from points to the nearest point on an isochrone.
    
    Parameters
    ----------
    points : array-like, shape (n, 2)
        Array of points with columns [color, magnitude]
    isochrone_color : array-like
        Color values of the isochrone
    isochrone_mag : array-like
        Magnitude values of the isochrone
    
    Returns
    -------
    distances : ndarray
        Euclidean distance of each point to the nearest isochrone point
    """
    return perpendicular_distance(points, isochrone_color, isochrone_mag)


def select_stars(star_color, star_mag, isochrone_color, isochrone_mag, 
                 threshold, metric='perpendicular'):
    """
    Select stars that lie within a specified distance from an isochrone.
    
    Parameters
    ----------
    star_color : array-like
        Color values of the stars (e.g., B-V, G-R)
    star_mag : array-like
        Magnitude values of the stars
    isochrone_color : array-like
        Color values of the isochrone
    isochrone_mag : array-like
        Magnitude values of the isochrone
    threshold : float
        Maximum distance from the isochrone for selection
    metric : str, optional
        Distance metric to use: 'perpendicular' or 'euclidean' (default: 'perpendicular')
    
    Returns
    -------
    mask : ndarray of bool
        Boolean mask indicating which stars are selected (True) or not (False)
    distances : ndarray
        Distances of each star from the isochrone
    """
    star_color = np.asarray(star_color)
    star_mag = np.asarray(star_mag)
    
    # Create array of star positions
    points = np.column_stack([star_color, star_mag])
    
    # Calculate distances based on metric
    if metric == 'perpendicular':
        distances = perpendicular_distance(points, isochrone_color, isochrone_mag)
    elif metric == 'euclidean':
        distances = euclidean_distance(points, isochrone_color, isochrone_mag)
    else:
        raise ValueError(f"Unknown metric: {metric}. Use 'perpendicular' or 'euclidean'.")
    
    # Create selection mask
    mask = distances <= threshold
    
    return mask, distances


def select_stars_color_range(star_color, star_mag, isochrone_color, isochrone_mag,
                              color_threshold, mag_threshold):
    """
    Select stars within a rectangular region around the isochrone.
    
    This method selects stars within ±color_threshold in color and ±mag_threshold
    in magnitude from the isochrone at each color value.
    
    Parameters
    ----------
    star_color : array-like
        Color values of the stars
    star_mag : array-like
        Magnitude values of the stars
    isochrone_color : array-like
        Color values of the isochrone
    isochrone_mag : array-like
        Magnitude values of the isochrone
    color_threshold : float
        Maximum color deviation from the isochrone
    mag_threshold : float
        Maximum magnitude deviation from the isochrone
    
    Returns
    -------
    mask : ndarray of bool
        Boolean mask indicating which stars are selected
    """
    star_color = np.asarray(star_color)
    star_mag = np.asarray(star_mag)
    isochrone_color = np.asarray(isochrone_color)
    isochrone_mag = np.asarray(isochrone_mag)
    
    # Ensure isochrone is sorted by color
    sort_idx = np.argsort(isochrone_color)
    isochrone_color = isochrone_color[sort_idx]
    isochrone_mag = isochrone_mag[sort_idx]
    
    # Interpolate isochrone magnitude as a function of color
    # Use bounds_error=False to handle stars outside the isochrone color range
    interp = interp1d(isochrone_color, isochrone_mag, kind='linear', 
                      bounds_error=False, fill_value=np.nan)
    
    # For each star, find the expected magnitude at its color
    expected_mag = interp(star_color)
    
    # Select stars within the color range
    color_mask = (star_color >= isochrone_color.min() - color_threshold) & \
                 (star_color <= isochrone_color.max() + color_threshold)
    
    # Select stars within magnitude threshold of the isochrone
    mag_mask = np.abs(star_mag - expected_mag) <= mag_threshold
    
    # Combine masks (handle NaN from interpolation)
    mask = color_mask & mag_mask & ~np.isnan(expected_mag)
    
    return mask
