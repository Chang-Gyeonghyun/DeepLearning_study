import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from data.dataloader import get_data_loaders

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_args():
    parser = argparse.ArgumentParser(description='PyTorch MNIST Training')
    parser.add_argument('--model', type=str, default='cnn',
                        help='choose which model to use: resnet or mobilenet (default: cnn)')
    args = parser.parse_args()
    return args

def import_model(model_name):
    module = __import__(f'models.{model_name}', fromlist=[model_name])
    return getattr(module, "Model")

def train_model(model_name):
    train_loader, _ = get_data_loaders()

    model_class = import_model(model_name)
    model = model_class().to(DEVICE)

    criterion  = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    epochs = 10

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
        print("Epoch : {:>4} / cost : {:>.9}".format(epoch + 1, loss))
        
    torch.save(model.state_dict(), './saved/mnist.pth')


if __name__ == "__main__":
    args = get_args()
    train_model(args.model)
