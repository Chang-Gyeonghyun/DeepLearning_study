import torch.nn as nn

class ConvNet(nn.Module):
  def __init__(self, layers):
    super(ConvNet, self).__init__()
    conv_layer = []
    fc_layer = []
    h, w, d = 32, 32, 3
    total_size = h*w*d

    for layer in layers:
      if layer.startswith('Conv'):
        out_channel = int(layer[4:])
        conv_layer += [nn.Conv2d(d, out_channel, kernel_size=3, padding=1),
                            nn.BatchNorm2d(out_channel),
                            nn.ReLU(inplace=True)]
        d = out_channel
        total_size = h*w*d
      elif layer.startswith('MaxPool'):
        conv_layer += [nn.MaxPool2d(2)]
        h, w = int(h/2.0), int(w/2.0)
        total_size = h*w*d
      elif layer.startswith('FC'):
        in_features = total_size
        out_features = int(layer[2:])
        fc_layer += [nn.Linear(in_features, out_features), nn.ReLU(inplace=True)]
        total_size = out_features

    fc_layer.pop()
    self.conv_layer = nn.Sequential(*conv_layer)
    self.fc_layer = nn.Sequential(*fc_layer)

  def forward(self, x):
    x = self.conv_layer(x)
    x = x.view(x.size(0), -1)
    x = self.fc_layer(x)
    return x