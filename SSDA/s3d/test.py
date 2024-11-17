import time
import numpy as np
import torch
import torch.nn as nn
import numpy as np
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from torch.utils.data import DataLoader

from utils import KDLoss, calc_mean_margin, weights_init, inv_lr_scheduler, adaptation_factor, cal_start_step_for_weight
from model import Predictor_deep, CustomResNet
from dataloader import get_dataset, return_stage2_dataset

def test(loader, G, F1, test=True):
    G.eval()
    F1.eval()
    test_loss = 0
    correct = 0
    size = 0
    criterion = nn.CrossEntropyLoss().cuda()

    with torch.no_grad():
        for batch_idx, (data_t, label_t) in enumerate(loader):
            data_t, label_t = data_t.to(device), label_t.to(device)
            feat = G(data_t) 
            output1 = F1(feat) 
            
            loss = criterion(output1, label_t)
            test_loss += loss.item()
            
            pred = output1.argmax(dim=1, keepdim=True)  
            correct += pred.eq(label_t.view_as(pred)).sum().item() 
            size += label_t.size(0)

    test_loss /= size
    accuracy = 100. * correct / size 
    if test:
        print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(test_loss, correct, size, accuracy))
    else:
        print('Validate set: Average loss: {:.4f}\n'.format(test_loss))
    
    return test_loss, accuracy


if __name__ == "__main__":
    args = {
        'lr': 0.01,
        'multi': 0.1,
        'batch_size': 32,
        'kd_lambda': 8,
        'temp': 4.0,
        'sty_layer': 'layer4',
        'sty_w': 0.1,

        'steps': 50000,
        'save_interval':1000,
        'log_interval': 100,
        'pseudo_interval': 100,
        'early': True,
        'patience': 5,
        # 바꿔야할 부분
        'checkpath': './saved/product-to-real_stage2.pth.tar',
        'alpha_value': 0.95
    }
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    source_dataset, _, _, _, _ = get_dataset()
    
    test_loader = DataLoader(source_dataset,
                                    batch_size=args['batch_size'] * 4, num_workers=2,
                                    shuffle=True, drop_last=True)

    G = CustomResNet()
    F1 = Predictor_deep()
    weights_init(F1)
    params = []
    for key, value in dict(G.named_parameters()).items():
        if value.requires_grad:
            if 'classifier' not in key:
                params += [{'params': [value], 'lr': args['multi'], 'weight_decay': 0.0005}]
            else:
                params += [{'params': [value], 'lr': args['multi'] * 10, 'weight_decay': 0.0005}]
    G.cuda()
    F1.cuda()

    G.train()
    F1.train()
    optimizer_g = optim.SGD(params, momentum=0.9, weight_decay=0.0005, nesterov=True)
    optimizer_f = optim.SGD(list(F1.parameters()), lr=0.01, momentum=0.9, weight_decay=0.0005, nesterov=True)
    def zero_grad_all():
        optimizer_g.zero_grad()
        optimizer_f.zero_grad()
    param_lr_g = [group['lr'] for group in optimizer_g.param_groups]
    param_lr_f = [group['lr'] for group in optimizer_f.param_groups]

    criterion = nn.CrossEntropyLoss().cuda()
    unl_criterion = nn.CrossEntropyLoss(reduction='none').cuda()
    kdloss = KDLoss(args['temp']).cuda()
    cos = nn.CosineSimilarity(dim=1, eps=1e-6).cuda()
    softmax = nn.Softmax(dim=1)
    all_step = args['steps']

    # 바꿔야할 부분
    checkpoint = torch.load('./saved/clip-to-real_stage2-papaer.pth.tar')
    G.load_state_dict(checkpoint['G'])
    F1.load_state_dict(checkpoint['F1'])
    optimizer_g.load_state_dict(checkpoint['optimizer_g'])
    optimizer_f.load_state_dict(checkpoint['optimizer_f'])
    start_step = checkpoint['step']

    acc_best = -np.inf
    counter = 0

    stage1_loss_test, stage1_acc_test = test(test_loader, G ,F1)
