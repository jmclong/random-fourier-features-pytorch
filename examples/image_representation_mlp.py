import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import rff
from itertools import repeat
from tqdm import tqdm


class MultiLayerBlock(nn.Module):
    def __init__(self, hidden_layer_size: int = 256):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(hidden_layer_size, hidden_layer_size),
            nn.ReLU()
        )

    def forward(self, x):
        return self.layers(x)


class ImageRepresentationMlp(nn.Module):
    def __init__(
            self, encoded_layer_size: int = 512, num_layers: int = 4,
            hidden_layer_size: int = 256):
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
    device = torch.device(
        'cuda') if torch.cuda.is_available() else torch.device('cpu')
    dataset = rff.dataloader.to_dataset('images/cat.jpg')
    encoding = rff.layers.GaussianEncoding(10.0, 2, 256).to(device)
    loss_fn = nn.MSELoss()
    network = ImageRepresentationMlp()
    network = network.to(device)
    optimizer = torch.optim.Adam(network.parameters(), lr=1e-2)
    X, y = dataset[:]
    X = X.to(device)
    Xp = encoding(X)
    y = y.to(device)
    for i in tqdm(range(100)):
        pred = network(Xp)
        loss = loss_fn(pred, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        coords = rff.dataloader.rectangular_coordinates(
            (512, 512)).to(device)
        coords = encoding(coords)
        image = network(coords)
        plt.imshow(image.cpu().numpy())
        plt.show()
