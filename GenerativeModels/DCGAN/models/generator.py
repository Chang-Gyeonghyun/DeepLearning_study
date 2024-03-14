import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, hidden=64):
        super().__init__()
        self.block = nn.Sequential(
            nn.ConvTranspose2d(100, hidden*8, 4, 1, 0, bias=False),
            nn.InstanceNorm2d(hidden*8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(hidden*8, hidden*4, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(hidden*4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(hidden*4, hidden*2, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(hidden*2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(hidden*2, hidden, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(hidden),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(hidden, 1, 1, 1, 2, bias=False),
            nn.Tanh()
        )

    def forward(self, x):
        return self.block(x)
