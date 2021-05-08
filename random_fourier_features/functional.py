import numpy as np
import torch

from torch import Tensor
from torch.types import _size


def sample_b(sigma: float, size: _size) -> Tensor:
    return torch.randn(size) * sigma


@torch.jit.script
def random_fourier_features_2d(
        v: Tensor,
        b: Tensor) -> Tensor:
    vp = 2 * np.pi * v @ b.T
    return torch.cat((torch.sin(vp), torch.cos(vp)), dim=-1)
