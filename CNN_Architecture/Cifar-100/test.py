import argparse
import torch
from torch.utils.data import DataLoader
from data.dataloader import get_data_loaders
import re
from models.resnet import ResidualBlock

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_args():
    parser = argparse.ArgumentParser(description='PyTorch MNIST Training')
    parser.add_argument('--model', type=str, default='cnn',
                        help='choose which model to use: resnet or mobilenet (default: cnn)')
    args = parser.parse_args()
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

def test_model(model_name):
    _, test_loader = get_data_loaders()

    model_class = import_model(model_name)
    if model_name=="cnn":
        model = model_class().to(DEVICE)
    elif re.match(r"efficientnet_b[0-6]", model_name):
        model = model_class(100).to(DEVICE)
    elif model_name=="mobilenet":
        model = model_class().to(DEVICE)
    elif model_name=="resnet":
        model = model_class(ResidualBlock, [3,4,6,3]).to(DEVICE)
        
    model.load_state_dict(torch.load(f'./saved/{model_name}.pth'))
    model.eval()
    correct = 0
    total = 0

    model.eval()
    with torch.no_grad():
        for image,label in test_loader:
            image = image.to(DEVICE)
            label= label.to(DEVICE)
            output = model(image)
            _,output_index = torch.max(output,1)
            total += label.size(0)
            correct += (output_index == label).sum().float()
        print("Accuracy of Test Data: {}%".format(100*correct/total))

if __name__ == "__main__":
    args = get_args()
    test_model(args.model)