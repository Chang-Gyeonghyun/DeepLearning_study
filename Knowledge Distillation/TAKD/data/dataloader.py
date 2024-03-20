import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader

def get_data_loaders(root="../../data/"):
    
    cifar_transform = transforms.Compose([transforms.ToTensor(),
                                      transforms.Normalize((0., 0., 0.), (0.5, 0.5, 0.5))])

    train_dataset = datasets.CIFAR10(root=root, train=True, download=True, transform=cifar_transform)
    test_dataset = datasets.CIFAR10(root=root, train=False, download=True, transform=cifar_transform)

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=True, num_workers=2)

    return train_loader, test_loader
