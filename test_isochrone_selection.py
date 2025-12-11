"""
Unit tests for isochrone_selection module.
"""

import unittest
import numpy as np
from isochrone_selection import (
    perpendicular_distance,
    euclidean_distance,
    select_stars,
    select_stars_color_range
)


class TestIsochroneSelection(unittest.TestCase):
    """Test cases for isochrone selection functions."""
    
    def setUp(self):
        """Set up test data."""
        # Simple linear isochrone
        self.iso_color = np.array([0.0, 1.0, 2.0, 3.0])
        self.iso_mag = np.array([10.0, 11.0, 12.0, 13.0])
        
        # Test stars: some on/near the isochrone, some far away
        self.star_color = np.array([0.5, 1.0, 1.5, 0.5, 2.5])
        self.star_mag = np.array([10.5, 11.0, 11.5, 15.0, 12.5])
    
    def test_perpendicular_distance_on_isochrone(self):
        """Test that points on the isochrone have zero distance."""
        points = np.column_stack([self.iso_color, self.iso_mag])
        distances = perpendicular_distance(points, self.iso_color, self.iso_mag)
        
        # Points on the isochrone should have approximately zero distance
        np.testing.assert_array_almost_equal(distances, np.zeros(len(points)), decimal=10)
    
    def test_perpendicular_distance_shapes(self):
        """Test that distance calculation returns correct shape."""
        points = np.column_stack([self.star_color, self.star_mag])
        distances = perpendicular_distance(points, self.iso_color, self.iso_mag)
        
        self.assertEqual(distances.shape, (len(self.star_color),))
        self.assertTrue(np.all(distances >= 0))
    
    def test_euclidean_distance(self):
        """Test Euclidean distance calculation."""
        points = np.column_stack([self.star_color, self.star_mag])
        distances = euclidean_distance(points, self.iso_color, self.iso_mag)
        
        self.assertEqual(distances.shape, (len(self.star_color),))
        self.assertTrue(np.all(distances >= 0))
    
    def test_select_stars_basic(self):
        """Test basic star selection with distance threshold."""
        threshold = 0.5
        mask, distances = select_stars(
            self.star_color, self.star_mag,
            self.iso_color, self.iso_mag,
            threshold=threshold
        )
        
        # Check shapes
        self.assertEqual(mask.shape, (len(self.star_color),))
        self.assertEqual(distances.shape, (len(self.star_color),))
        
        # Check that mask is boolean
        self.assertEqual(mask.dtype, bool)
        
        # Check that selected stars have distance <= threshold
        self.assertTrue(np.all(distances[mask] <= threshold))
        
        # Check that unselected stars have distance > threshold
        self.assertTrue(np.all(distances[~mask] > threshold))
    
    def test_select_stars_all_selected(self):
        """Test selection with very large threshold (all stars selected)."""
        threshold = 1000.0
        mask, distances = select_stars(
            self.star_color, self.star_mag,
            self.iso_color, self.iso_mag,
            threshold=threshold
        )
        
        # All stars should be selected with large threshold
        self.assertTrue(np.all(mask))
    
    def test_select_stars_none_selected(self):
        """Test selection with very small threshold (no stars selected)."""
        threshold = 0.001
        mask, distances = select_stars(
            self.star_color, self.star_mag,
            self.iso_color, self.iso_mag,
            threshold=threshold
        )
        
        # Expect very few or no stars selected
        # At least the stars far from the isochrone should not be selected
        self.assertFalse(mask[3])  # Star at (0.5, 15.0) is far from isochrone
    
    def test_select_stars_invalid_metric(self):
        """Test that invalid metric raises ValueError."""
        with self.assertRaises(ValueError):
            select_stars(
                self.star_color, self.star_mag,
                self.iso_color, self.iso_mag,
                threshold=0.5,
                metric='invalid_metric'
            )
    
    def test_select_stars_color_range_basic(self):
        """Test color-magnitude range selection."""
        color_threshold = 0.5
        mag_threshold = 0.5
        
        mask = select_stars_color_range(
            self.star_color, self.star_mag,
            self.iso_color, self.iso_mag,
            color_threshold=color_threshold,
            mag_threshold=mag_threshold
        )
        
        # Check shape and type
        self.assertEqual(mask.shape, (len(self.star_color),))
        self.assertEqual(mask.dtype, bool)
    
    def test_select_stars_color_range_boundaries(self):
        """Test that stars outside color range are not selected."""
        # Create stars clearly outside the color range
        star_color_out = np.array([-2.0, 10.0])
        star_mag_out = np.array([10.0, 10.0])
        
        mask = select_stars_color_range(
            star_color_out, star_mag_out,
            self.iso_color, self.iso_mag,
            color_threshold=0.1,
            mag_threshold=0.5
        )
        
        # Stars far outside color range should not be selected
        self.assertFalse(np.any(mask))
    
    def test_empty_inputs(self):
        """Test handling of empty input arrays."""
        empty_color = np.array([])
        empty_mag = np.array([])
        
        mask, distances = select_stars(
            empty_color, empty_mag,
            self.iso_color, self.iso_mag,
            threshold=0.5
        )
        
        self.assertEqual(len(mask), 0)
        self.assertEqual(len(distances), 0)
    
    def test_single_star(self):
        """Test selection with a single star."""
        single_color = np.array([1.0])
        single_mag = np.array([11.0])
        
        mask, distances = select_stars(
            single_color, single_mag,
            self.iso_color, self.iso_mag,
            threshold=0.5
        )
        
        self.assertEqual(len(mask), 1)
        self.assertTrue(mask[0])  # Star at (1.0, 11.0) is on the isochrone
    
    def test_unsorted_isochrone(self):
        """Test that function handles unsorted isochrone data."""
        # Create unsorted isochrone
        unsorted_color = np.array([2.0, 0.0, 3.0, 1.0])
        unsorted_mag = np.array([12.0, 10.0, 13.0, 11.0])
        
        # Should not raise an error
        mask, distances = select_stars(
            self.star_color, self.star_mag,
            unsorted_color, unsorted_mag,
            threshold=0.5
        )
        
        self.assertEqual(len(mask), len(self.star_color))


class TestIsochroneDistanceMetrics(unittest.TestCase):
    """Test specific distance metric properties."""
    
    def test_distance_symmetry(self):
        """Test that distance is symmetric (not dependent on point order)."""
        iso_color = np.linspace(0, 1, 10)
        iso_mag = np.linspace(10, 11, 10)
        
        # Two identical points
        points1 = np.array([[0.5, 10.5], [0.3, 10.3]])
        points2 = np.array([[0.3, 10.3], [0.5, 10.5]])
        
        dist1 = perpendicular_distance(points1, iso_color, iso_mag)
        dist2 = perpendicular_distance(points2, iso_color, iso_mag)
        
        # Distances should be the same regardless of order
        np.testing.assert_array_almost_equal(dist1[[1, 0]], dist2)
    
    def test_distance_monotonicity(self):
        """Test that distance increases as we move away from isochrone."""
        iso_color = np.array([0.5, 0.5, 0.5])
        iso_mag = np.array([10.0, 11.0, 12.0])
        
        # Points at increasing distance from isochrone
        points = np.array([
            [0.5, 11.0],   # On isochrone
            [0.6, 11.0],   # Slightly off
            [0.8, 11.0],   # Further off
            [1.5, 11.0],   # Even further
        ])
        
        distances = perpendicular_distance(points, iso_color, iso_mag)
        
        # Distances should increase
        self.assertLess(distances[0], distances[1])
        self.assertLess(distances[1], distances[2])
        self.assertLess(distances[2], distances[3])


if __name__ == '__main__':
    unittest.main()
