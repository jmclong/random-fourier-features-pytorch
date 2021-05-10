import numpy as np
import torch
import unittest
import rff


class LayerTest(unittest.TestCase):

    def test_random_fourier_features(self):
        b = rff.functional.sample_b(1.0, (256, 2))
        layer = rff.layers.RandomFourierFeatures(b=b)
        v = rff.dataloader.rectangular_coordinates((256, 256))
        gamma_v = layer(v)
        gamma_v_expected = rff.functional.random_fourier_features(v, b)
        np.testing.assert_almost_equal(gamma_v.numpy(), gamma_v_expected.numpy(), decimal=5)
