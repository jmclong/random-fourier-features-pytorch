import numpy as np
import unittest
import rff


class LayerTest(unittest.TestCase):
    """
    Test layers, the point to to make sure they can be created and that the output matches
    the corresponding functional calls. Other testing will be performed in the functional
    test.
    """

    def test_gaussian_encoding(self):
        b = rff.functional.sample_b(1.0, (256, 2))
        layer = rff.layers.GaussianEncoding(b=b)
        v = rff.dataloader.rectangular_coordinates((256, 256))
        gamma_v = layer(v)
        gamma_v_expected = rff.functional.gaussian_encoding(v, b)
        np.testing.assert_almost_equal(
            gamma_v.numpy(),
            gamma_v_expected.numpy(),
            decimal=5)

    def test_basic_encoding(self):
        layer = rff.layers.BasicEncoding()
        v = rff.dataloader.rectangular_coordinates((256, 256))
        gamma_v = layer(v)
        gamma_v_expected = rff.functional.basic_encoding(v)
        np.testing.assert_almost_equal(
            gamma_v.numpy(),
            gamma_v_expected.numpy(),
            decimal=5)

    def test_positional_encoding(self):
        layer = rff.layers.PositionalEncoding(sigma=1.0, m=10)
        v = rff.dataloader.rectangular_coordinates((256, 256))
        gamma_v = layer(v)
        gamma_v_expected = rff.functional.positional_encoding(
            v, sigma=1.0, m=10)
        np.testing.assert_almost_equal(
            gamma_v.numpy(),
            gamma_v_expected.numpy(),
            decimal=5)
