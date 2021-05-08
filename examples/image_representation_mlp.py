import torch.nn as nn
import random_fourier_features.layers as rff

from itertools import repeat


class MultiLayerBlock(nn.Module):
    def __init__(self, hidden_layer_size: float = 256):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(hidden_layer_size, hidden_layer_size),
            nn.ReLU()
        )


class ImageRepresentationMlp(nn.Module):
    def __init__(
            self, encoded_layer_size: float = 512, num_layers: float = 4,
            hidden_layer_size: float = 256):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(encoded_layer_size, hidden_layer_size),
            *repeat(MultiLayerBlock(hidden_layer_size), num_layers),
            nn.Linear(hidden_layer_size, 3),
            nn.Sigmoid()
        )
