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

    source_dataset, labeled_target_dataset, unlabeled_target_dataset, val_dataset, test_dataset = get_dataset()
    
    val_loader = DataLoader(val_dataset,
                            batch_size=args['batch_size'] * 4, num_workers=2,
                            shuffle=True, drop_last=True)
    test_loader = DataLoader(test_dataset,
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
    checkpoint = torch.load('./saved/product-to-real.pth.tar')
    G.load_state_dict(checkpoint['G'])
    F1.load_state_dict(checkpoint['F1'])
    optimizer_g.load_state_dict(checkpoint['optimizer_g'])
    optimizer_f.load_state_dict(checkpoint['optimizer_f'])
    start_step = checkpoint['step']

    acc_best = -np.inf
    counter = 0

    stage1_loss_test, stage1_acc_test = test(test_loader, G ,F1)
    first_lambda_step = cal_start_step_for_weight(args['kd_lambda'], stage1_acc_test)
    stage2_margin = calc_mean_margin(G, F1)

    for step in range(start_step, all_step):
        optimizer_g = inv_lr_scheduler(param_lr_g, optimizer_g, step, init_lr=args['lr'])
        optimizer_f = inv_lr_scheduler(param_lr_f, optimizer_f, step, init_lr=args['lr'])
        lr = optimizer_f.param_groups[0]['lr']

        lambda_step = first_lambda_step + (step - start_step)
        lamb = adaptation_factor(lambda_step * 1.0 / 50000, args['kd_lambda'])

        interval = args['pseudo_interval']
        if (step % interval == 0) or (step == start_step):
            train_loader = return_stage2_dataset(args, G, F1, stage2_margin)
            data_iter_train = iter(train_loader)
        data_train = next(data_iter_train)

        zero_grad_all()

        data = data_train[0].to(device)
        target = data_train[1].to(device)
        batch_size = data.size(0)
        lab_data = data[:batch_size // 2]
        lab_target = target[:batch_size // 2]
        unl_data = data[batch_size // 2:]
        lab_feature, x_sty = G.forward_mean_var(lab_data, args['sty_layer'])
        lab_output = F1(lab_feature)
        lab_loss = criterion(lab_output, lab_target)

        # kd_loss
        unl_feature, assistant_feature = G.forward_assistant(unl_data, x_sty, args['sty_w'], args['sty_layer'])
        unlabeled_output = F1(unl_feature)
        assistant_output = F1(assistant_feature)
        kd_loss = kdloss(unlabeled_output, assistant_output.detach())

        with torch.no_grad():
            unl_pred = softmax(unlabeled_output)
            max_unl_pred = torch.max(unl_pred, dim=1)[0]
            pseudo_label = torch.max(unl_pred, dim=1)[1]
        unl_loss = unl_criterion(unlabeled_output, pseudo_label.detach())
        weighted_unl_loss = torch.mul(max_unl_pred.detach(), unl_loss)
        mean_unl_loss = torch.mean(weighted_unl_loss)
        # sum all the loss
        loss = lab_loss + (lamb * kd_loss) + mean_unl_loss

        # backward
        loss.backward()
        optimizer_g.step()
        optimizer_f.step()

        G.zero_grad()
        F1.zero_grad()
        zero_grad_all()

        if step % args['log_interval'] == 0:
            print('Train Ep: {}\tlr: {}\t' \
                    'Loss Classification: {:.6f}\tUNL Loss Classification: {:.6f}\tLoss KD: {:.6f}\tLoss total: {:.6f}' \
                    .format(step, lr, lab_loss.data, mean_unl_loss.data, kd_loss.data, loss.data), end='\t')
            
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
