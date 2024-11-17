import math
import torch
import torch.nn as nn
from torch.autograd import Function
import torch.utils
import torch.utils.data
from dataloader import get_dataset

def inv_lr_scheduler(param_lr, optimizer, iter_num, gamma=0.0001,
                     power=0.75, init_lr=0.001):
    lr = init_lr * (1 + gamma * iter_num) ** (- power)
    i = 0
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr * param_lr[i]
        i += 1
    return optimizer

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.1)
    elif classname.find('Linear') != -1:
        nn.init.xavier_normal_(m.weight)
        nn.init.zeros_(m.bias)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.1)
        m.bias.data.fill_(0)

class GradReverse(Function):
    def __init__(self, lambd):
        self.lambd = lambd

    def forward(self, x):
        return x.view_as(x)

    def backward(self, grad_output):
        return (grad_output * -self.lambd)


def grad_reverse(x, lambd=1.0):
    return GradReverse(lambd)(x)

def adaptation_factor(x, kd_lambda):
    if x >= 1.0:
        return 1.0
    den = 1.0 + math.exp(-kd_lambda * x)
    lamb = 2.0 / den - 1.0
    return lamb


def cal_start_step_for_weight(kd_lambda, stage1_acc):
    v_acc = stage1_acc * 1.5
    v_acc = v_acc / 100
    if v_acc >= 1.0:
        return 100000
    lambda_step = - (50000 / kd_lambda) * math.log(2/(1 + v_acc) - 1)
    lambda_step = int(lambda_step)
    return lambda_step

class KDLoss(nn.Module):
    def __init__(self, temp_factor):
        super(KDLoss, self).__init__()
        self.temp_factor = temp_factor
        self.kl_div = nn.KLDivLoss(reduction = "sum")

    def forward(self, input, target):
        log_p = torch.log_softmax(input/self.temp_factor, dim=1)
        q = torch.softmax(target/self.temp_factor, dim=1)
        loss = self.kl_div(log_p, q)*(self.temp_factor**2)/input.size(0)
        return loss
    
def calc_mean_margin(G, F1):

    G.eval()
    F1.eval()
    _, _, unlabeled_target_dataset, _, _ = get_dataset()
    batch_size = 800
    total_margin = 0
    target_loader_unl = torch.utils.data.DataLoader(unlabeled_target_dataset,
                                                    batch_size=batch_size, num_workers=2,
                                                    shuffle=False, drop_last=False)

    for i, (img, label) in enumerate(target_loader_unl):
        img = img.cuda()

        with torch.no_grad():
            feature = G(img)
            output = F1(feature)

        top_two_class_prob = torch.topk(output, 2)[0]
        for j in range(len(label)):
            cal_margin = top_two_class_prob[j,0] - top_two_class_prob[j,1]
            total_margin = total_margin + cal_margin

    mean_margin = total_margin * 1.0 / len(target_loader_unl)

    G.train()
    F1.train()

    return mean_margin