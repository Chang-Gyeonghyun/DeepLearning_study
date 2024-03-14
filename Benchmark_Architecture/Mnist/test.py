import argparse
import torch
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

def test_model(model_name):
    _, test_loader = get_data_loaders()
    model_class = import_model(model_name)
    model = model_class().to(DEVICE)
    model.load_state_dict(torch.load('./saved/mnist.pth'))
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