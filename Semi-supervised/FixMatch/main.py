import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from dataloader import get_cifar10
import numpy as np
from utils import get_cosine_schedule_with_warmup, initial_model, get_data_loader

def evaluate(args, test_loader, model, epoch):
    model.eval()
    test_loss = 0
    correct_top1 = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(args['DEVICE']), target.to(args['DEVICE'])
            output = model(data)
            test_loss += F.cross_entropy(output, target).item()

            _, output_index = torch.max(output, 1)
            correct_top1 += (output_index == target).sum().item()

        test_loss /= len(test_loader.dataset)
        top1_acc = 100. * correct_top1 / len(test_loader.dataset)

        print('\nEpochs: {} Test set: Average loss: {:.4f}, Top-1 Accuracy: {}/{} ({:.8f}%)\n'.format(
            epoch, test_loss, correct_top1, len(test_loader.dataset), top1_acc))

    return top1_acc

def eval_model_update(train_model, eval_model, ema_m):
    for param_train, param_eval in zip(train_model.parameters(), eval_model.parameters()):
        param_eval.copy_(param_eval * ema_m + param_train.detach() * (1-ema_m))
    
    for buffer_train, buffer_eval in zip(train_model.buffers(), eval_model.buffers()):
        buffer_eval.copy_(buffer_train)        

def train(args, loader_dict, model, eval_model, optimizer, scheduler):
    total_sup_loss = 0
    total_unsup_loss = 0
    total_loss_sum = 0
    total_util_ratio = 0
    num_iters = min(args['num_iters'], len(loader_dict['train_lb']))
    
    for i, ((x_lb, y_lb), (x_ulb_w, x_ulb_s, _)) in enumerate(zip(loader_dict['train_lb'], loader_dict['train_ulb'])):
        if i >= args['num_iters']:
            break
        num_lb = y_lb.shape[0]
        inputs = torch.cat((x_lb, x_ulb_w, x_ulb_s)).to(args['DEVICE'])
        outputs = model(inputs)
        logits_x_lb = outputs[:num_lb]
        logits_x_ulb_w, logits_x_ulb_s = outputs[num_lb:].chunk(2)

        sup_loss = F.cross_entropy(logits_x_lb, y_lb.to(args['DEVICE'], dtype=torch.long), reduction='mean')
        probs_x_ulb_w = torch.softmax(logits_x_ulb_w.detach(), dim=-1)

        probs_x_ulb = probs_x_ulb_w.detach()
        max_probs, _ = torch.max(probs_x_ulb, dim=-1)
        mask = max_probs.ge(args['threshold']).to(max_probs.dtype)
        pseudo_label = torch.argmax(probs_x_ulb, dim=-1)

        unsup_loss = F.cross_entropy(logits_x_ulb_s, pseudo_label, reduction='none') * mask
        unsup_loss = unsup_loss.mean() 
        total_loss = sup_loss + args['wu'] * unsup_loss

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        scheduler.step()
        
        with torch.no_grad():
            eval_model_update(model, eval_model, args['ema_m'])
        del inputs

        total_sup_loss += sup_loss.item()
        total_unsup_loss += unsup_loss.item()
        total_loss_sum += total_loss.item()
        total_util_ratio += mask.float().mean().item()

    avg_sup_loss = total_sup_loss / num_iters
    avg_unsup_loss = total_unsup_loss / num_iters
    avg_total_loss = total_loss_sum / num_iters
    avg_util_ratio = total_util_ratio / num_iters

    print(f"Average supervised loss: {avg_sup_loss:.4f}")
    print(f"Average unsupervised loss: {avg_unsup_loss:.4f}")
    print(f"Average total loss: {avg_total_loss:.4f}")
    print(f"Average utilization ratio: {avg_util_ratio:.4f}")


if __name__ == "__main__":
    
    args={}
    args['wu'] = 1
    args['threshold'] = 0.95
    args['DEVICE'] = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args['epochs'] = 1024
    args['batch_size'] = 64
    args['lr'] = 0.03
    args['momentum'] = 0.9
    args['weight_decay'] = 5e-4
    args['num_iters'] = 1024
    args['uratio'] = 7
    args['num_label'] = 4000
    args['ema_m'] = 0.999
    args['data_dir'] = "../data"
    args['train_sampler'] = 'RandomSampler'

    train_labeled_dataset, train_unlabeled_dataset, test_dataset = get_cifar10(args, '../data/')

    train_labeled_loader = get_data_loader(train_labeled_dataset, batch_size=args['batch_size'], num_iters=args['num_iters'])
    train_unlabeled_loader = get_data_loader(train_unlabeled_dataset, batch_size=args['batch_size']*args['uratio'], num_iters=args['num_iters'])
    test_loader = DataLoader(test_dataset, batch_size=args['batch_size'], num_workers=2, pin_memory=True, shuffle=False)
    loader_dict={}
    loader_dict['train_lb'] = train_labeled_loader
    loader_dict['train_ulb'] = train_unlabeled_loader
    loader_dict['test'] = test_loader

    # train_dset = SSL_Dataset(train=True, 
    #                          num_classes=10, data_dir=args['data_dir'])
    # lb_dset, ulb_dset = train_dset.get_ssl_dset(args['num_label'])
    
    # _eval_dset = SSL_Dataset(train=False, 
    #                          num_classes=10, data_dir=args['data_dir'])
    # eval_dset = _eval_dset.get_dset()
    
    # loader_dict = {}
    # dset_dict = {'train_lb': lb_dset, 'train_ulb': ulb_dset, 'eval': eval_dset}
    
    # loader_dict['train_lb'] = get_data_loader(dset_dict['train_lb'],
    #                                           args['batch_size'],
    #                                           data_sampler = args['train_sampler'],
    #                                           num_iters=args['num_iters'],
    #                                           num_workers=2)
    
    # loader_dict['train_ulb'] = get_data_loader(dset_dict['train_ulb'],
    #                                            args['batch_size']*args['uratio'],
    #                                            data_sampler = args['train_sampler'],
    #                                            num_iters=args['num_iters'],
    #                                            num_workers=4*2)
    
    # loader_dict['eval'] = get_data_loader(dset_dict['eval'],
    #                                       args['batch_size'], 
    #                                       num_workers=2)

    model, eval_model = initial_model()

    model = model.to(args['DEVICE'])
    eval_model = eval_model.to(args['DEVICE']) 
    optimizer = torch.optim.SGD(model.parameters(), lr=args['lr'], momentum=args['momentum'], 
                                weight_decay=args['weight_decay'])
    scheduler = get_cosine_schedule_with_warmup(optimizer, args['num_iters'])

    best_acc = -np.inf
    for epoch in range(args['epochs']):
        train(args, loader_dict, model, eval_model, optimizer, scheduler)

        val_acc = evaluate(args, loader_dict['eval'] , model, epoch)

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict() , "saved/BestModel.pth")