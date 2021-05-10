import numpy as np
import torch
import unittest
import rff.functional as rffF


class FunctionalTest(unittest.TestCase):

    def test_sample_b(self):
        b = rffF.sample_b(1.0, (3, 4))
        self.assertEqual(b.shape, (3, 4))

    def test_random_fourier_features(self):
        v = torch.eye(2)
        b = torch.eye(2)
        gamma_v = rffF.random_fourier_features(v, b)
        self.assertEqual(gamma_v.shape, (2, 4))
        xc = np.cos(2 * np.pi * np.eye(2))
        yc = np.sin(2 * np.pi * np.eye(2))
        gamma_v_expected = np.concatenate((xc, yc), axis=-1)
        np.testing.assert_almost_equal(gamma_v, gamma_v_expected, decimal=5)
