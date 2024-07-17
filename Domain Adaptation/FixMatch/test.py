import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
from dataloader import get_cifar10
from model import WideResNet
import numpy as np

def test(args, test_loader, model, criterion, epoch):
    model.eval()
    test_loss = 0
    correct_top1 = 0
    correct_top5 = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(args['DEVICE']), target.to(args['DEVICE'])
            output = model(data)
            test_loss += criterion(output, target).item()

            # Top-1 정확도 계산
            _, output_index = torch.max(output, 1)
            correct_top1 += (output_index == target).sum().item()

            # Top-5 정확도 계산
            _, top5_indices = output.topk(5, dim=1)
            correct_top5 += (top5_indices == target.view(-1, 1)).sum().item()

        test_loss /= len(test_loader.dataset)
        top1_acc = 100. * correct_top1 / len(test_loader.dataset)
        top5_acc = 100. * correct_top5 / len(test_loader.dataset)

        print('\nEpochs: {} Test set: Average loss: {:.4f}, Top-1 Accuracy: {}/{} ({:.8f}%), Top-5 Accuracy: {}/{} ({:.8f}%)\n'.format(epoch,
            test_loss, correct_top1, len(test_loader.dataset), top1_acc,
            correct_top5, len(test_loader.dataset), top5_acc))

    return top1_acc

if __name__ == "__main__":
    args={}
    args['DEVICE'] = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_labeled_dataset, train_unlabeled_dataset, test_dataset = get_cifar10('../../data/')
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=True, num_workers=2)

    model = WideResNet(depth=28, num_classes=10, widen_factor=10, drop_rate=0.3)
    model.load_state_dict(torch.load('saved/BestModel.pth', map_location=torch.device('cpu')))
    model = model.to(args['DEVICE'])
    criterion  = nn.CrossEntropyLoss()
    val_acc = test(args, test_loader , model, criterion, 1)