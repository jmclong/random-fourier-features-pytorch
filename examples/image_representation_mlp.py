import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import rff
from itertools import repeat
from torch.utils.data import DataLoader
from tqdm import tqdm

class MultiLayerBlock(nn.Module):
    def __init__(self, hidden_layer_size: float = 256):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(hidden_layer_size, hidden_layer_size),
            nn.ReLU()
        )
    
    def forward(self, x):
        return torch.relu(x + self.layers(x))


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
    
    def forward(self, x):
        return self.layers(x)


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    dataset = rff.dataloader.to_dataset('examples/images/cat.jpg')
    loss_fn  = nn.L1Loss()
    dataloader = DataLoader(dataset, batch_size=2048, pin_memory=True, num_workers=12)

    network = nn.Sequential(
        rff.layers.RandomFourierFeatures(10.0, 2, 256),
        ImageRepresentationMlp()
    )
    network = network.to(device)

    optimizer = torch.optim.Adam(network.parameters(), lr=1e-4)
    size = len(dataloader.dataset)
    for _ in range(100):
        for batch, (X, y) in enumerate(tqdm(dataloader)):
            X = X.cuda(non_blocking=True)
            y = y.cuda(non_blocking=True)
            pred = network(X)
            loss = loss_fn(pred, y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
    with torch.no_grad():
        coords = rff.dataloader.rectangular_coordinates((958, 2087))
        network = network.cpu()
        image = network(coords)
        plt.imshow(image.numpy())
        plt.show()