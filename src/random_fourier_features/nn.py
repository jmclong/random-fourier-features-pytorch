import torch
import torch.nn as nn

from typing import Optional

import random_fourier_features.functional as rF


class RandomFourierFeatures2d(nn.Module):
    def __init__(self, sigma: float, b: Optional[torch.Tensor] = None):
        if b is None:
            self.b = nn.parameter.Parameter(
                rF.sample_b(sigma), requires_grad=False)
        else:
            self.b = b

    def forward(self, v):
        return rF.random_fourier_features_2d(v, self.b)
