import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self, hidden=64):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(1, hidden, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(hidden, hidden*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(hidden*2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(hidden*2, hidden*4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(hidden*4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(hidden*4, 1, 4, 2, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.block(x).view(-1, 1).squeeze(1)
