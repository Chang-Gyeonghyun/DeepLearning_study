import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.fc_layer = nn.Sequential(
            nn.Linear(in_features=64*7*7, out_features=100, bias=True),
            nn.ReLU(),
            nn.Linear(in_features=100, out_features=10, bias=True)
        )

    def forward(self, x):
        x = self.layer(x)
        x = x.view(x.shape[0], -1)
        x = self.fc_layer(x)
        return x
