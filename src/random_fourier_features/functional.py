import numpy as np
import torch


def sample_b(sigma: float, **kwargs) -> torch.Tensor:
    return torch.randn(**kwargs) * sigma


def random_fourier_features_2d(
        v: torch.Tensor,
        b: torch.Tensor) -> torch.Tensor:
    vp = 2 * np.pi * v @ b.T
    return torch.cat((torch.sin(vp), torch.cos(vp)), dim=-1)
