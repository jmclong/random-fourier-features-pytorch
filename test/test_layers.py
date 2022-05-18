"""
Test layers, the point to to make sure they can be created and that the output matches
the corresponding functional calls. Other testing will be performed in the functional
test.
"""

import numpy as np
import rff
import pytest
import torch


def check_cuda(device):
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('Cuda is not available')


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_gaussian_encoding(device):
    check_cuda(device)
    b = rff.functional.sample_b(1.0, (256, 2)).to(device)
    layer = rff.layers.GaussianEncoding(b=b).to(device)
    v = rff.dataloader.rectangular_coordinates((256, 256)).to(device)
    gamma_v = layer(v)
    gamma_v_expected = rff.functional.gaussian_encoding(v, b)
    np.testing.assert_almost_equal(
        gamma_v.cpu().numpy(),
        gamma_v_expected.cpu().numpy(),
        decimal=5)


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_basic_encoding(device):
    check_cuda(device)
    layer = rff.layers.BasicEncoding().to(device)
    v = rff.dataloader.rectangular_coordinates((256, 256)).to(device)
    gamma_v = layer(v)
    gamma_v_expected = rff.functional.basic_encoding(v)
    np.testing.assert_almost_equal(
        gamma_v.cpu().numpy(),
        gamma_v_expected.cpu().numpy(),
        decimal=5)


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_positional_encoding(device):
    check_cuda(device)
    layer = rff.layers.PositionalEncoding(sigma=1.0, m=10).to(device)
    v = rff.dataloader.rectangular_coordinates((256, 256)).to(device)
    gamma_v = layer(v)
    gamma_v_expected = rff.functional.positional_encoding(
        v, sigma=1.0, m=10)
    np.testing.assert_almost_equal(
        gamma_v.cpu().numpy(),
        gamma_v_expected.cpu().numpy(),
        decimal=5)
