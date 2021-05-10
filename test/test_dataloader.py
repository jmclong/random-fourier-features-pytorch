import numpy as np
import unittest
import rff.dataloader as rffd


class DataloaderTest(unittest.TestCase):

    def test_rectangular_coordinates_shape(self):
        size = (3, 4, 5)
        coords = rffd.rectangular_coordinates(size)
        self.assertEqual(coords.shape, (np.prod(size), len(size)))

    def test_rectangular_coordinates_shape(self):
        size = (2, 2)
        coords = rffd.rectangular_coordinates(size)
        expected_coords = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        np.testing.assert_equal(coords, expected_coords)
