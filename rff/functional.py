import numpy as np
import torch

from torch import Tensor

def sample_b(sigma: float, size: tuple) -> Tensor:
    r"""Matrix of size :attr:`size` sampled from from :math:`\mathcal{N}(0, \sigma^2)`

    Args:
        sigma (float): standard deviation
        size (tuple) size of the matrix sampled

    See :class:`~rff.layers.RandomFourierFeatures2d` for more details
    """
    return torch.randn(size) * sigma


@torch.jit.script
def random_fourier_features(
        v: Tensor,
        b: Tensor) -> Tensor:
    r"""Computes :math:`\gamma(\mathbf{v}) = (\cos{2 \pi \mathbf{B} \mathbf{v}} , \sin{2 \pi \mathbf{B} \mathbf{v}})`

    Args:
        v (Tensor): input tensor of shape :math:`(\text{minibatch}, \text{input_size})`
        b (Tensor): projection matrix of shape :math:`(\text{encoded_layer_size}, \text{input_size})`

    Returns:
        Tensor: mapped tensor of shape :math:`(\text{minibatch}, 2 \cdot \text{encoded_layer_size})`

    See :class:`~rff.layers.RandomFourierFeatures2d` for more details.
    """
    vp = 2 * np.pi * v @ b.T
    return torch.cat((torch.cos(vp), torch.sin(vp)), dim=-1)
