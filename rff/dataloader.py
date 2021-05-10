import torch

from torch import Tensor

def rectangular_coordinates(size: tuple) -> Tensor:
    r"""Creates a tensor

    Args:
        size (tuple): [description]

    Returns:
        Tensor: [description]
    """
    linspace_func = lambda nx: torch.linspace(0, 1, nx)
    linspaces = map(linspace_func, size)
    coordinates = torch.meshgrid(*linspaces)
    flatten_func = lambda xc: torch.flatten(xc)
    flattened = map(flatten_func, coordinates)
    return torch.stack(tuple(flattened), axis=-1)
