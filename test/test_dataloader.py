import numpy as np
import unittest
import rff


class DataloaderTest(unittest.TestCase):

    def test_rectangular_coordinates_shape(self):
        size = (3, 4, 5)
        coords = rff.dataloader.rectangular_coordinates(size)
        self.assertEqual(coords.shape, (*size, len(size)))

    def test_rectangular_coordinates_value(self):
        size = (2, 2)
        coords = rff.dataloader.rectangular_coordinates(size)
        expected_coords = np.array([[[0, 0], [0, 1]], [[1, 0], [1, 1]]])
        np.testing.assert_equal(coords, expected_coords)

    def test_to_dataset(self):
        dataset = rff.dataloader.to_dataset('images/cat.jpg')
