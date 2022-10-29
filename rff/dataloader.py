import torch
import torchvision

from torch import Tensor
from torch.utils.data.dataset import TensorDataset


def rectangular_coordinates(size: tuple) -> Tensor:
    r"""Creates a tensor of equally spaced coordinates for use with an image or volume

    Args:
        size (tuple): shape of the image or volume

    Returns:
        Tensor: tensor of shape :math:`(*\text{size}, \text{len(size)})`
    """
    def linspace_func(nx): return torch.linspace(0.0, 1.0, nx)
    linspaces = map(linspace_func, size)
    coordinates = torch.meshgrid(*linspaces, indexing='ij')
    return torch.stack(coordinates, dim=-1)


def to_dataset(path: str) -> TensorDataset:
    image = torchvision.io.read_image(path).float()
    _, H, W = image.shape
    coords = rectangular_coordinates((H, W))
    image = image.permute((1, 2, 0))
    image /= 255.0
    coords = coords.flatten(0, -2)
    image = image.flatten(0, -2)
    return TensorDataset(coords, image)
