import argparse
import torch
import torch.nn as nn
from data.dataloader import get_data_loaders
from models.resnet import ResidualBlock
import re

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_args():
    parser = argparse.ArgumentParser(description='PyTorch Cifar-100 Training')
    parser.add_argument('--model', type=str, default='cnn',
                        help='choose which model to use: resnet or mobilenet (default: cnn)')
    args = parser.se_args()
    return args

def import_model(model_name):
    if re.match(r"efficientnet_b[0-6]", model_name):
        version = model_name[-1]
        model_name = model_name[:-3]
    module = __import__(f'models.{model_name}', fromlist=[model_name])
    if model_name=="cnn":
        return getattr(module, "Model")
    elif model_name=="efficientnet":
        return getattr(module, f"efficientnet_b{version}")
    elif model_name=="mobilenet":
        return getattr(module, "MobileNet")
    elif model_name=="resnet":
        return getattr(module, "ResNet")

def train_model(model_name):
    train_loader, _ = get_data_loaders()

    model_class = import_model(model_name)
    if model_name=="cnn":
        model = model_class().to(DEVICE)
    elif re.match(r"efficientnet_b[0-6]", model_name):
        model = model_class(100).to(DEVICE)
    elif model_name=="mobilenet":
        model = model_class().to(DEVICE)
    elif model_name=="resnet":
        model = model_class(ResidualBlock, [3,4,6,3]).to(DEVICE)

    criterion  = nn.CrossEntropyLoss().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer=optimizer,
                                        lr_lambda=lambda epoch: 0.95 ** (epoch // 10),
                                        last_epoch=-1,
                                        verbose=False)
    epochs = 100
    print("Starting Training Loop...")
    for epoch in range(epochs):
        model.train()
        train_correct = 0
        for image, label in train_loader:
            image = image.to(DEVICE)
            label = label.to(DEVICE)
            optimizer.zero_grad()
            output = model(image)
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()
            _,output_index = torch.max(output,1)
            train_correct += (output_index == label).sum().float()
        scheduler.step()
        print("Epoch : {:>4} / cost : {:>.9}".format(epoch + 1, loss))
        
    torch.save(model.state_dict(), f'./saved/{model_name}.pth')


if __name__ == "__main__":
    args = get_args()
    train_model(args.model)
