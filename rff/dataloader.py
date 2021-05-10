import torch

from torch import Tensor


def rectangular_coordinates(size: tuple) -> Tensor:
    r"""Creates a tensor of equally spaced coordinates for use with something like and image or volume

    Args:
        size (tuple): shape of the image or volume

    Returns:
        Tensor: tensor of shape :math:`(\text{minibatch}, \prod_i \text{size}_i)`
    """
    def linspace_func(nx): return torch.linspace(0, 1, nx)
    linspaces = map(linspace_func, size)
    coordinates = torch.meshgrid(*linspaces)
    def flatten_func(xc): return torch.flatten(xc)
    flattened = map(flatten_func, coordinates)
    return torch.stack(tuple(flattened), axis=-1)
