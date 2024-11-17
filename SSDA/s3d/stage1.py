import time
import numpy as np
import torch
import torch.nn as nn
import numpy as np
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from torch.utils.data import DataLoader

from utils import weights_init, inv_lr_scheduler
from model import Predictor_deep, CustomResNet
from dataloader import get_dataset

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

        'steps': 50000,
        'save_interval':1000,
        'log_interval': 100,
        'early': True,
        'patience': 5,
        'checkpath': './saved/product-to-clip.pth.tar'

    }
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    source_dataset, labeled_target_dataset, unlabeled_target_dataset, val_dataset, test_dataset = get_dataset()

    source_loader = DataLoader(source_dataset, batch_size=args['batch_size'] * 2,
                                                num_workers=2, shuffle=True,
                                                drop_last=True)
    labeled_target_loader = DataLoader(labeled_target_dataset,
                                    batch_size=args['batch_size'],
                                    num_workers=2,
                                    shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset,
                            batch_size=args['batch_size'] * 4, num_workers=2,
                            shuffle=True, drop_last=True)
    test_loader = DataLoader(test_dataset,
                            batch_size=args['batch_size'] * 4, num_workers=2,
                            shuffle=True, drop_last=True)

    G = CustomResNet()
    F1 = Predictor_deep()
    weights_init(F1)
    G.cuda()
    F1.cuda()

    params = []
    for key, value in dict(G.named_parameters()).items():
        if value.requires_grad:
            if 'classifier' not in key:
                params += [{'params': [value], 'lr': args['multi'], 'weight_decay': 0.0005}]
            else:
                params += [{'params': [value], 'lr': args['multi'] * 10, 'weight_decay': 0.0005}]

    optimizer_g = optim.SGD(params, momentum=0.9, weight_decay=0.0005, nesterov=True)
    optimizer_f = optim.SGD(list(F1.parameters()), lr=0.01, momentum=0.9, weight_decay=0.0005, nesterov=True)
    def zero_grad_all():
        optimizer_g.zero_grad()
        optimizer_f.zero_grad()
    param_lr_g = [group['lr'] for group in optimizer_g.param_groups]
    param_lr_f = [group['lr'] for group in optimizer_f.param_groups]

    criterion = nn.CrossEntropyLoss().cuda()

    all_step = args['steps']

    data_iter_s = iter(source_loader)
    data_iter_t = iter(labeled_target_loader)
    len_train_source = len(source_loader)
    len_train_target = len(labeled_target_loader)

    acc_best = -np.inf
    counter = 0

    G.train()
    F1.train()
    for step in range(all_step):
        optimizer_g = inv_lr_scheduler(param_lr_g, optimizer_g, step, init_lr=args['lr'])
        optimizer_f = inv_lr_scheduler(param_lr_f, optimizer_f, step, init_lr=args['lr'])
        lr = optimizer_f.param_groups[0]['lr']

        if step % len_train_source == 0:
            data_iter_s = iter(source_loader)
        if step % len_train_target == 0:
            data_iter_t = iter(labeled_target_loader)
        data_s = next(data_iter_s)
        data_t = next(data_iter_t)
            
        data = torch.cat((data_s[0], data_t[0])).to(device)
        label = torch.cat((data_s[1], data_t[1])).to(device)

        zero_grad_all()
        
        feature = G(data)
        output = F1(feature)
        loss = criterion(output, label)
        loss.backward()
        optimizer_g.step()
        optimizer_f.step()
        zero_grad_all()

        G.zero_grad()
        F1.zero_grad()
        zero_grad_all()

        if step % args['log_interval'] == 0:
            print('Train iter: {}\tlr: {}\tLoss Classification: {:.6f}'. \
            format(step, lr, loss.data), end='\t')
            loss_val, acc_val = test(val_loader, G, F1, False)

        if step % args['save_interval'] == 0 and step > 0:
            loss_test, acc_test = test(test_loader, G, F1)
            G.train()
            F1.train()
            if acc_val >= acc_best:
                acc_best = acc_val
                counter = 0
                torch.save({
                    'G': G.state_dict(),
                    'F1': F1.state_dict(),
                    'optimizer_g': optimizer_g.state_dict(),
                    'optimizer_f': optimizer_f.state_dict(),
                    'step': step + 1,
                }, args['checkpath'])
            else:
                counter += 1
            if args['early']:
                if counter > args['patience']:
                    break
