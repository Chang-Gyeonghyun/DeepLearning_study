import torch.nn as nn

class ResidualBlock(nn.Module):
  def __init__(self, in_channels, middle_channel, stride=1, down_sample=False):
    super(ResidualBlock, self).__init__()
    self.relu = nn.ReLU(inplace=True)
    self.layer1 = nn.Sequential(
        nn.Conv2d(in_channels, middle_channel, kernel_size=1, bias=False),
        nn.BatchNorm2d(middle_channel),
        nn.ReLU(inplace=True))

    self.layer2 = nn.Sequential(
        nn.Conv2d(middle_channel, middle_channel, kernel_size=3,
                         stride=stride, padding=1, bias=False),
        nn.BatchNorm2d(middle_channel),
        nn.ReLU(inplace=True))

    self.layer3 = nn.Sequential(
        nn.Conv2d(middle_channel, middle_channel*4, kernel_size=1,bias=False),
        nn.BatchNorm2d(middle_channel*4))

    if down_sample:
      self.down_sample = nn.Sequential(
                nn.Conv2d(in_channels, middle_channel*4, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(middle_channel*4),
            )
    else:
      self.down_sample = None


  def forward(self, x):
    shortcut = x

    out = self.layer1(x)
    out = self.layer2(out)
    out = self.layer3(out)

    if self.down_sample is not None:
      shortcut = self.down_sample(x)

    out += shortcut
    out = self.relu(out)
    return out

class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=10):
        super(ResNet, self).__init__()
        self.inplanes = 64
        # 첫 번째 레이어
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3,bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # 레이어 블록 생성
        self.layer1 = self._make_layer(block, 64, layers[0], 1)
        self.layer2 = self._make_layer(block, 128, layers[1], 2)
        self.layer3 = self._make_layer(block, 256, layers[2], 2)
        self.layer4 = self._make_layer(block, 512, layers[3], 2)

        # 마지막 fully connected layer
        self.avgpool = nn.AvgPool2d(1, stride=1)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(2048, num_classes)

    def _make_layer(self, block, middle_planes, blocks, stride=1):
        strides = [stride] + [1] * (blocks-1)
        layers = []
        layers.append(block(self.inplanes, middle_planes, strides[0], True))
        self.inplanes = middle_planes * 4
        for i in range(1, blocks):
            layers.append(block(self.inplanes, middle_planes, strides[i]))
        return nn.Sequential(*layers)

    def forward(self, x):
      x = self.conv1(x)
      x = self.bn1(x)
      x = self.relu(x)
      x = self.maxpool(x)
      x = self.layer1(x)
      x = self.layer2(x)
      x = self.layer3(x)
      x = self.layer4(x)
      x = self.avgpool(x)
      x = self.fc(self.flatten(x))
      return x

def ResNet50():
    return ResNet(ResidualBlock, [3, 4, 6, 3])

def ResNet101():
    return ResNet(ResidualBlock, [3, 4, 23, 3])