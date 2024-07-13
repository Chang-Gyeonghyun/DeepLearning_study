import torch

def test(args, test_loader, model, criterion, epoch):
    model.eval()
    test_loss = 0
    correct_top1 = 0
    correct_top5 = 0
    with torch.no_grad():
        for data, target in test_loader:
            if args['cuda']:
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