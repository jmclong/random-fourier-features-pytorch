import torch

from torch import Tensor
from torch.types import _size

def coords2d(size: _size) -> Tensor:
    nx, ny = size
    x = torch.linspace(0, 1, nx)
    y = torch.linspace(0, 1, ny)
    xc, yc = torch.meshgrid(x, y)
    xf = torch.flatten(xc)
    yf = torch.flatten(yc)
    return torch.stack((xf, yf), axis=-1)