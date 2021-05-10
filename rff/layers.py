import torch.nn as nn

from typing import Optional
from torch import Tensor
import rff.functional as rffF


class RandomFourierFeatures(nn.Module):
    def __init__(self, sigma: float, input_size: float,
                 hidden_size: float, b: Optional[Tensor] = None):
        r"""Computes :math:`\gamma(\mathbf{v}) = (\cos{2 \pi \mathbf{B} \mathbf{v}} , \sin{2 \pi \mathbf{B} \mathbf{v}})`

        Args:
            sigma (float): standard deviation for 
            input_size (float): size of the 
            hidden_size (float): [description]
            b (Optional[Tensor], optional): Optionally specify a :attr:`b` matrix already sampled
        """
        super().__init__()
        if b is None:
            self.b = nn.parameter.Parameter(
                rffF.sample_b(sigma, (hidden_size, input_size)),
                requires_grad=False)
        else:
            self.b = nn.parameter.Parameter(b, requires_grad=False)

    def forward(self, v: Tensor) -> Tensor:
        r"""Maps regular coordinates using random Fourier features

        Args:
            v (Tensor): Tensor of regular coordinates of size :math:`(\text{minibatch}, \text{input_size})`

        Returns:
            Tensor: [description]
        """
        return rffF.random_fourier_features(v, self.b)
