import torch.nn as nn

from typing import Optional
from torch.types import _size
from torch import Tensor
import random_fourier_features.functional as rffF


class RandomFourierFeatures2d(nn.Module):
    def __init__(self, sigma: float, input_size: float,
                 hidden_size: float, b: Optional[Tensor] = None):
        super().__init__()
        if b is None:
            self.b = nn.parameter.Parameter(
                rffF.sample_b(sigma, (hidden_size, input_size)),
                requires_grad=False)
        else:
            self.b = nn.parameter.Parameter(b, requires_grad=False)

    def forward(self, v):
        return rffF.random_fourier_features_2d(v, self.b)
