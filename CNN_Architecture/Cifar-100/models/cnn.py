import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),

            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),

            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),

            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
        )
        self.fc_layer = nn.Sequential(
            nn.Linear(in_features=128*4*4, out_features=512, bias=True),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Linear(in_features=512, out_features=256, bias=True),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Linear(in_features=256, out_features=128, bias=True),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Linear(in_features=128, out_features=100, bias=True)
        )

    def forward(self, x):
        for i in range(1, 4):
            layer = getattr(self, f'layer{i}')
            x = layer(x)
            x = F.max_pool2d(x, kernel_size=2, stride=2)

        x = x.view(x.shape[0], -1)
        x = self.fc_layer(x)
        return x