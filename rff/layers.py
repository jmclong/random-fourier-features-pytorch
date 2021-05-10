import torch.nn as nn

from typing import Optional
from torch import Tensor
import rff.functional as rffF


class RandomFourierFeatures(nn.Module):
    """Layer for mapping coordinates using random Fourier features"""

    def __init__(self, sigma: float, input_size: float,
                 encoded_size: float, b: Optional[Tensor] = None):
        r"""Computes :math:`\gamma(\mathbf{v}) = (\cos{2 \pi \mathbf{B} \mathbf{v}} , \sin{2 \pi \mathbf{B} \mathbf{v}})`

        Args:
            sigma (float): standard deviation
            input_size (float): the number of input dimensions
            encoded_size (float): the number of dimensions the `b` matrix maps to
            b (Optional[Tensor], optional): Optionally specify a :attr:`b` matrix already sampled
        """
        super().__init__()
        if b is None:
            b = rffF.sample_b(sigma, (encoded_size, input_size))
        self.b = nn.parameter.Parameter(b, requires_grad=False)

    def forward(self, v: Tensor) -> Tensor:
        r"""Maps regular coordinates using random Fourier features

        Args:
            v (Tensor): Tensor of regular coordinates of size :math:`(\text{minibatch}, \text{input_size})`

        Returns:
            Tensor: Tensor mapping using random fourier features of shape :math:`(\text{minibatch}, \text{2 \cdot encoded_size})`
        """
        return rffF.random_fourier_features(v, self.b)
