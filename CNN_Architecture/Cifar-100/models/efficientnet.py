import torch.nn as nn
import torch.nn.functional as F
import torch

class Swish(nn.Module):
    def __init__(self):
        super().__init__()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return x + self.sigmoid(x)
    
class SEBlock(nn.Module):
    def __init__(self, in_channels, r):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool2d((1,1))
        self.excitation = nn.Sequential(
            nn.Conv2d(in_channels = in_channels, out_channels = in_channels*4, kernel_size = 1, stride = 1),
            Swish(),
            nn.Conv2d(in_channels = in_channels*4, out_channels = in_channels, kernel_size = 1, stride = 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        out = self.squeeze(x)
        out = self.excitation(out)
        return x * out
    
class Depthwise(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super().__init__()
        padding = 0
        if kernel_size == 3:
            padding = 1
        elif kernel_size == 5:
            padding = 2

        self.depthwise = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size, stride=stride, padding=padding, groups=in_channels, bias=False),
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
    
class MBConv1(nn.Module):
  def __init__(self, in_channels, out_channels, kernel_size, stride, p=0.5):
    super(MBConv1, self).__init__()
    self.p = torch.tensor(p).float() if (in_channels == out_channels) else torch.tensor(1).float()
    self.in_channels = in_channels
    self.out_channels = out_channels
    self.stride = stride
    self.block = nn.Sequential(
        Depthwise(in_channels = in_channels,
                        out_channels = in_channels,
                        kernel_size = kernel_size,
                        stride = stride),
        nn.BatchNorm2d(in_channels),
        Swish(),

        SEBlock(in_channels = in_channels, r = 4),

        nn.Conv2d(in_channels = in_channels,
                out_channels = out_channels,
                kernel_size = 1,
                stride = 1),
        nn.BatchNorm2d(out_channels)
    )
    self.shortcut = (stride == 1) and (in_channels == out_channels)

  def forward(self, x):
    if self.training:
        if not torch.bernoulli(self.p):
            return x
    x_shortcut = x
    x = self.block(x)

    if self.shortcut:
        x= x_shortcut + x

    return x

class MBConv6(nn.Module):
  def __init__(self, in_channels, out_channels, kernel_size, stride, p=0.5):
    super(MBConv6, self).__init__()
    self.p = torch.tensor(p).float() if (in_channels == out_channels) else torch.tensor(1).float()
    self.in_channels = in_channels
    self.out_channels = out_channels
    self.stride = stride

    self.block = nn.Sequential(
      nn.Conv2d(in_channels = in_channels, out_channels = 6 * in_channels, kernel_size = 1),
      nn.BatchNorm2d(6*in_channels),
      Swish(),

      Depthwise(in_channels = 6*in_channels, out_channels= 6*in_channels, kernel_size = kernel_size, stride = stride),
      nn.BatchNorm2d(num_features = 6*in_channels),
      Swish(),

      SEBlock(in_channels = 6 * in_channels, r = 4),
      nn.Conv2d(in_channels = 6*in_channels, out_channels = out_channels, kernel_size = 1),

      nn.BatchNorm2d(out_channels)
    )
    self.shortcut = (stride == 1) and (in_channels == out_channels)

  def forward(self, x):
    if self.training:
        if not torch.bernoulli(self.p):
            return x
    x_shortcut = x
    x = self.block(x)

    if self.shortcut:
        x= x_shortcut + x

    return x

class EfficientNet(nn.Module):
  def __init__(self, num_classes, width_coef=1., depth_coef=1., scale=1., stochastic_depth=False, p=0.5):
    super(EfficientNet, self).__init__()
    channels = [32, 16, 24, 40, 80, 112, 192, 320, 1280]
    depth = [1, 2, 2, 3, 3, 4, 1]
    self.scale = scale

    channels = [int(x*width_coef) for x in channels]
    depth = [int(x*depth_coef) for x in depth]
    self.in_channels = 3
    if stochastic_depth:
        self.p = p
        self.step = (1 - 0.5) / (sum(depth) - 1)
    else:
        self.p = 1
        self.step = 0

    self.stage_1 = nn.Conv2d(self.in_channels, channels[0], 3, stride=2, padding=1)

    self.stage_2 = [MBConv1(channels[0], channels[1], 3, 1, p)]
    for _ in range(depth[0] - 1):
      self.stage_2.append(MBConv1(channels[1], channels[1], 3, 1, p))
    self.stage_2 = nn.Sequential(*self.stage_2)

    self.stage_3 = [MBConv6(channels[1], channels[2], 3, 2, p)]
    for _ in range(depth[1] - 1):
      self.stage_3.append(MBConv6(channels[2], channels[2], 3, 1, p))
    self.stage_3 = nn.Sequential(*self.stage_3)

    self.stage_4 = [MBConv6(channels[2], channels[3], 5, 2, p)]
    for _ in range(depth[2] - 1):
      self.stage_4.append(MBConv6(channels[3], channels[3], 5, 1, p))
    self.stage_4 = nn.Sequential(*self.stage_4)

    self.stage_5 = [MBConv6(channels[3], channels[4], 3, 2, p)]
    for _ in range(depth[3] - 1):
      self.stage_5.append(MBConv6(channels[4], channels[4], 3, 1, p))
    self.stage_5 = nn.Sequential(*self.stage_5)

    self.stage_6 = [MBConv6(channels[4], channels[5], 5, 1, p)]
    for _ in range(depth[4] - 1):
      self.stage_6.append(MBConv6(channels[5], channels[5], 5, 1, p))
    self.stage_6 = nn.Sequential(*self.stage_6)

    self.stage_7 = [MBConv6(channels[5], channels[6], 5, 2, p)]
    for _ in range(depth[5] - 1):
      self.stage_7.append(MBConv6(channels[6], channels[6], 5, 1, p))
    self.stage_7 = nn.Sequential(*self.stage_7)

    self.stage_8 = [MBConv6(channels[6], channels[7], 3, 1, p)]
    for _ in range(depth[6] - 1):
      self.stage_8.append(MBConv6(channels[7], channels[7], 3, 1, p))
    self.stage_8 = nn.Sequential(*self.stage_8)

    self.stage_9 = nn.Sequential(
        nn.Conv2d(channels[7], channels[8], 1, 1),
        nn.AdaptiveMaxPool2d(1),
        nn.Flatten(),
        nn.Linear(channels[8], num_classes)
    )

  def forward(self, x):
    self.in_channels = x.shape[1]
    x = F.interpolate(x, scale_factor=self.scale, mode='bilinear', align_corners=False)

    for i in range(1, 10):
        stage_name = "stage_{}".format(i)
        x = getattr(self, stage_name)(x)
        self.p -= self.step

    return x
  
def efficientnet_b0(num_classes=1000):
    return EfficientNet(num_classes=num_classes, width_coef=1.0, depth_coef=1.0, scale=1.0, stochastic_depth = True, p=0.2)

def efficientnet_b1(num_classes=1000):
    return EfficientNet(num_classes=num_classes, width_coef=1.0, depth_coef=1.1, scale=240/224, stochastic_depth = True, p=0.2)

def efficientnet_b2(num_classes=1000):
    return EfficientNet(num_classes=num_classes, width_coef=1.1, depth_coef=1.2, scale=260/224., stochastic_depth = True, p=0.3)

def efficientnet_b3(num_classes=1000):
    return EfficientNet(num_classes=num_classes, width_coef=1.2, depth_coef=1.4, scale=300/224, stochastic_depth = True, p=0.3)

def efficientnet_b4(num_classes=1000):
    return EfficientNet(num_classes=num_classes, width_coef=1.4, depth_coef=1.8, scale=380/224, stochastic_depth = True, p=0.4)

def efficientnet_b5(num_classes=1000):
    return EfficientNet(num_classes=num_classes, width_coef=1.6, depth_coef=2.2, scale=456/224, stochastic_depth = True, p=0.4)

def efficientnet_b6(num_classes=1000):
    return EfficientNet(num_classes=num_classes, width_coef=1.8, depth_coef=2.6, scale=528/224, stochastic_depth = True, p=0.5)

def efficientnet_b7(num_classes=1000):
    return EfficientNet(num_classes=num_classes, width_coef=2.0, depth_coef=3.1, scale=600/224, stochastic_depth = True, p=0.5)