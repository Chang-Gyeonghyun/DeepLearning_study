import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
from torch.distributions.beta import Beta
from utils import grad_reverse



class Predictor_deep(nn.Module):
    def __init__(self, num_class=65, inc=512, temp=0.05):
        super(Predictor_deep, self).__init__()
        self.fc1 = nn.Linear(inc, 512)
        self.fc2 = nn.Linear(512, num_class, bias=False)
        self.num_class = num_class
        self.temp = temp

    def forward(self, x, reverse=False, eta=0.1):
        x = self.fc1(x)
        if reverse:
            x = grad_reverse(x, eta)
        x = F.normalize(x)
        x_out = self.fc2(x) / self.temp
        return x_out

class CustomResNet(models.ResNet):
    def __init__(self, pretrained=True, num_classes=1000):
        super(CustomResNet, self).__init__(models.resnet.BasicBlock, [3, 4, 6, 3], num_classes=num_classes)
        
        if pretrained:
            state_dict = models.resnet34(pretrained=True).state_dict()
            self.load_state_dict(state_dict)
        
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.maxpool = nn.Identity()
        self.fc = nn.Identity()

    def forward_mean_var(self, x, sty_layer):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x1_cm = self.gap2d(x, keepdims=True)
        x1_cv = self.var2d(x, keepdims=True)

        x = self.layer2(x)
        if sty_layer in ['layer2', 'layer3', 'layer4']:
            x2_cm = self.gap2d(x, keepdims=True)
            x2_cv = self.var2d(x, keepdims=True)
        else:
            x2_cm = 0
            x2_cv = 0

        x = self.layer3(x)
        if sty_layer in ['layer3', 'layer4']:
            x3_cm = self.gap2d(x, keepdims=True)
            x3_cv = self.var2d(x, keepdims=True)
        else:
            x3_cm = 0
            x3_cv = 0

        x = self.layer4(x)
        if sty_layer in ['layer4']:
            x4_cm = self.gap2d(x, keepdims=True)
            x4_cv = self.var2d(x, keepdims=True)
        else:
            x4_cm = 0
            x4_cv = 0

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x_sty = [(x1_cm, x1_cv), (x2_cm, x2_cv), (x3_cm, x3_cv), (x4_cm, x4_cv)]

        return x, x_sty

    def forward_assistant(self, x, x_sty, sty_w, sty_layer):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x1_cm = self.gap2d(x, keepdims=True)
        x1_cv = self.var2d(x, keepdims=True)
        assist_x = self.styletransform_detach(x, x1_cm, x1_cv, x_sty[0][0], x_sty[0][1], sty_w)

        x = self.layer2(x)
        assist_x = self.layer2(assist_x)
        if sty_layer in ['layer2', 'layer3', 'layer4']:
            x2_cm = self.gap2d(assist_x, keepdims=True)
            x2_cv = self.var2d(assist_x, keepdims=True)
            assist_x = self.styletransform_detach(assist_x, x2_cm, x2_cv, x_sty[1][0], x_sty[1][1], sty_w)

        x = self.layer3(x)
        assist_x = self.layer3(assist_x)
        if sty_layer in ['layer3', 'layer4']:
            x3_cm = self.gap2d(assist_x, keepdims=True)
            x3_cv = self.var2d(assist_x, keepdims=True)
            assist_x = self.styletransform_detach(assist_x, x3_cm, x3_cv, x_sty[2][0], x_sty[2][1], sty_w)

        x = self.layer4(x)
        assist_x = self.layer4(assist_x)
        if sty_layer in ['layer4']:
            x4_cm = self.gap2d(assist_x, keepdims=True)
            x4_cv = self.var2d(assist_x, keepdims=True)
            assist_x = self.styletransform_detach(assist_x, x4_cm, x4_cv, x_sty[3][0], x_sty[3][1], sty_w)

        x = self.avgpool(x)
        assist_x = self.avgpool(assist_x)
        x = x.view(x.size(0), -1)
        assist_x = assist_x.view(assist_x.size(0), -1)
        return x, assist_x

    def styletransform_detach(self, x, x_m, x_v, y_m, y_v, sty_w):
        x_m, x_v, y_m, y_v = x_m.detach(), x_v.detach(), y_m.detach(), y_v.detach()
        eps = 1e-6

        batch_size = x.size(0)
        if sty_w == 0.75:
            lmda = torch.full((batch_size, 1, 1, 1), 0.75).to(x.device)
        elif sty_w == 0.5:
            lmda = torch.full((batch_size, 1, 1, 1), 0.5).to(x.device)
        elif sty_w == 0.25:
            lmda = torch.full((batch_size, 1, 1, 1), 0.25).to(x.device)
        else:
            lmda = Beta(sty_w, sty_w).sample((x.size(0), 1, 1, 1)).to(x.device)

        x_v = (x_v + eps).sqrt()
        y_v = (y_v + eps).sqrt()

        sty_mean = lmda * y_m + (1 - lmda) * x_m
        sty_std = lmda * y_v + (1 - lmda) * x_v

        sty_x = sty_std * ((x - x_m) / x_v) + sty_mean
        return sty_x

    def gap2d(self, x, keepdims=False):
        out = torch.mean(x.view(x.size(0), x.size(1), -1), -1)
        if keepdims:
            out = out.view(out.size(0), out.size(1), 1, 1)
        return out

    def var2d(self, x, keepdims=False):
        out = torch.var(x.view(x.size(0), x.size(1), -1), -1)
        if keepdims:
            out = out.view(out.size(0), out.size(1), 1, 1)
        return out