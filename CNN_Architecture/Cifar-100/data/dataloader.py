# data/dataloader.py

import torch
from torch.utils.data import DataLoader, Dataset, TensorDataset, ConcatDataset
from torchvision import transforms
from utils.unpickle import unpickle

def get_cifar_transform():
    return transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 평균 0.5, 표준 편차 0.5로 정규화
    ])

def apply_augmentation(train_images, train_labels, cifar_transform, num_augmentations=2):
    augmented_images = []
    augmented_labels = []
    
    for _ in range(num_augmentations):
        for i in range(len(train_images)):
            img = transforms.ToPILImage()(train_images[i])
            img = cifar_transform(img)
            augmented_images.append(img)
            augmented_labels.append(train_labels[i])
    
    return augmented_images, augmented_labels

def get_data_loaders(root="../../data/", batch_size=128, num_workers=2):
    train_data = unpickle(f"{root}cifar-100-python/train")
    test_data = unpickle(f"{root}cifar-100-python/test")
    meta = unpickle(f"{root}/cifar-100-python/meta")

    train_images = torch.tensor(train_data[b'data'], dtype=torch.float32)
    train_labels = torch.tensor(train_data[b'fine_labels'], dtype=torch.long)
    test_images = torch.tensor(test_data[b'data'], dtype=torch.float32)
    test_labels = torch.tensor(test_data[b'fine_labels'], dtype=torch.long)

    train_images = train_images.view(-1,3,32,32)
    test_images = test_images.view(-1,3,32,32)
    
    train_images = (train_images / 255.0 - 0.5) / 0.5
    test_images = (test_images / 255.0 - 0.5) / 0.5
    
    cifar_transform = get_cifar_transform()
    
    augmented_images, augmented_labels = apply_augmentation(train_images, train_labels, cifar_transform)
    augmented_dataset = TensorDataset(torch.stack(augmented_images), torch.tensor(augmented_labels))
    train_dataset = TensorDataset(train_images, train_labels)
    test_dataset = TensorDataset(test_images, test_labels)
    combined_dataset = ConcatDataset([train_dataset, augmented_dataset])

    train_loader = DataLoader(combined_dataset, batch_size=128, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=True, num_workers=2)

    return train_loader, test_loader
