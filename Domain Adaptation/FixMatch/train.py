import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
from dataloader import get_cifar10
from model import WideResNet
from test import test
import numpy as np

def train(args, train_labeled_loader, train_unlabeled_loader, model, optimizer, lr_scheduler, epoch):
    model.train()
    epoch_loss = 0

    labeled_iter = iter(train_labeled_loader)
    unlabeled_iter = iter(train_unlabeled_loader)

    max_iterations = max(len(train_labeled_loader), len(train_unlabeled_loader))
    labeled_epoch, unlabeled_epoch = 0, 0
    for batch_idx in range(max_iterations):

        try:
            labeled_batch = next(labeled_iter)
            labeled_epoch += 1
        except StopIteration:
            labeled_epoch = 0
            labeled_iter = iter(train_labeled_loader)
            labeled_batch = next(labeled_iter)

        try:
            unlabeled_batch = next(unlabeled_iter)
            unlabeled_epoch += 1
        except StopIteration:
            unlabeled_epoch = 0
            unlabeled_iter = iter(train_unlabeled_loader)
            unlabeled_batch = next(unlabeled_iter)
        x_weak, labels = labeled_batch
        (u_weak, u_strong), _ = unlabeled_batch

        optimizer.zero_grad()
        inputs = torch.cat((x_weak, u_weak, u_strong)).to(args['DEVICE'])
        labels = labels.to(args['DEVICE'])

        logits = model(inputs)
        logits_x = logits[:len(x_weak)]
        logits_u_weak, logits_u_strong = logits[len(x_weak):].chunk(2)

        labeled_loss = F.cross_entropy(logits_x, labels)

        with torch.no_grad():
            pseudo_labels = torch.softmax(logits_u_weak, dim=1)
            max_probs, targets_u = torch.max(pseudo_labels, dim=1)
            mask = torch.where(max_probs >= args['threshold'], torch.tensor(1.0).to(args['DEVICE']), torch.tensor(0.0).to(args['DEVICE']))

        unlabeled_loss = (F.cross_entropy(logits_u_strong, targets_u, reduction="none") * mask).mean()

        loss = labeled_loss.mean() + args['wu'] * unlabeled_loss

        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    if lr_scheduler:
        lr_scheduler.step()

    print('\nEpoch: {} Train set: Average loss: {:.4f}'.format(epoch, epoch_loss / max_iterations))

if __name__ == "__main__":
    
    args={}
    args['wu'] = 1
    args['threshold'] = 0.95
    args['DEVICE'] = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args['epochs'] = 100

    train_labeled_dataset, train_unlabeled_dataset, test_dataset = get_cifar10('../../data/')

    train_labeled_loader = DataLoader(train_labeled_dataset, batch_size=128, shuffle=True, num_workers=2)
    train_unlabeled_loader = DataLoader(train_unlabeled_dataset, batch_size=128, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=True, num_workers=2)

    model = WideResNet(depth=28, num_classes=10, widen_factor=10, drop_rate=0.3)
    model = model.to(args['DEVICE'])
    criterion  = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.03)
    Cosine_lr_scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=220)

    best_acc = -np.inf

    for epoch in range(args['epochs']):
        train(args, train_labeled_loader, train_unlabeled_loader, model,
                                    optimizer, Cosine_lr_scheduler, epoch)

        val_acc = test(args, test_loader , model, criterion, epoch)

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict() , "saved/BestModel.pth")