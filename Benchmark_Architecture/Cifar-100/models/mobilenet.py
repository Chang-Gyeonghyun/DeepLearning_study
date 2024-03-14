import torch
import torch.nn as nn

class Depthwise(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        self.depthwise = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, stride=stride, padding=1, groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
        )

        self.pointwise = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
    
    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x
    
class MobileNet(nn.Module):
    def __init__(self, num_class=100):
        super().__init__()
        self.ConvBasic = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        self.ConvDW = nn.Sequential(
            Depthwise(32, 64, 1),
            Depthwise(64, 128, 2),
            Depthwise(128, 128, 1),
            Depthwise(128, 256, 2),
            Depthwise(256, 256, 1),
            Depthwise(256, 512, 1),
            Depthwise(512, 512, 1),
            Depthwise(512, 512, 1),
            Depthwise(512, 512, 1),
            Depthwise(512, 512, 1),
            Depthwise(512, 512, 1),
            Depthwise(512, 1024, 2),
            Depthwise(1024, 1024, 1),
        )
        self.fc_layer = nn.Linear(1024, num_class)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        x = self.ConvBasic(x)
        x = self.ConvDW(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layer(x)
        return x
