import numpy as np
from torchvision import transforms, datasets
from torch.utils.data import Dataset
from PIL import Image

class TransformFixMatch(object):
    def __init__(self, mean, std):
        self.weak = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=32,
                                  padding=int(32*0.125),
                                  padding_mode='reflect')])
        self.strong = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=32,
                                  padding=int(32*0.125),
                                  padding_mode='reflect'),
            transforms.RandAugment(num_ops=2, magnitude=9)])
        self.normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)])

    def __call__(self, x):
        weak = self.weak(x)
        strong = self.strong(x)
        return self.normalize(weak), self.normalize(strong)
    
class CustomDataset(Dataset):

    def __init__(self, x_data, y_data, transform=None):
        self.x_data = x_data
        self.y_data = y_data
        self.transform = transform
        self.len = len(x_data)

    def __getitem__(self, index):
        image = self.x_data[index]
        label = self.y_data[index]

        if self.transform:
          image = Image.fromarray(image)
          if self.transform==TransformFixMatch:
              w_image, s_image = self.transform(image)

              return w_image, s_image, label
          image = self.transform(image)

        return image, label

    def __len__(self):
        return self.len
    
def unlabeled_data_split(data, labels, num_class, num_data):
    num_data_per_class = num_data // num_class

    labeled_data = []
    unlabeled_data = []
    labeled_target = []
    unlabeled_target = []

    labels = np.array(labels)
    data = np.array(data)

    for class_ in range(num_class):
        idx = np.where(labels == class_)[0]
        shuffled_idx = np.random.choice(idx, len(idx), replace=False)

        labeled_data.append(data[shuffled_idx[:num_data_per_class]])
        unlabeled_data.append(data[shuffled_idx[num_data_per_class:]])

        labeled_target.append(labels[shuffled_idx[:num_data_per_class]])
        unlabeled_target.append(labels[shuffled_idx[num_data_per_class:]])

    labeled_data = np.concatenate(labeled_data, axis=0)
    unlabeled_data = np.concatenate(unlabeled_data, axis=0)
    labeled_target = np.concatenate(labeled_target, axis=0)
    unlabeled_target = np.concatenate(unlabeled_target, axis=0)

    return labeled_data, unlabeled_data, labeled_target, unlabeled_target

cifar10_mean = (0.4914, 0.4822, 0.4465)
cifar10_std = (0.2471, 0.2435, 0.2616)

def get_cifar10(root):
    transform_labeled = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(size=32,padding=int(32*0.125),padding_mode='reflect'),
        transforms.ToTensor(),
        transforms.Normalize(mean=cifar10_mean, std=cifar10_std)
    ])

    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=cifar10_mean, std=cifar10_std)
    ])

    train_dataset = datasets.CIFAR10(root, train=True, download=True)

    labeled_data, unlabeled_data, labeled_target, unlabeled_target = \
              unlabeled_data_split(train_dataset.data, train_dataset.targets, 10, 4000)

    train_labeled_dataset = CustomDataset(labeled_data, labeled_target, transform_labeled)
    train_unlabeled_dataset = CustomDataset(unlabeled_data, unlabeled_target, TransformFixMatch(mean=cifar10_mean, std=cifar10_std))

    test_dataset = datasets.CIFAR10(
        root, train=False, transform=transform_val, download=True)

    return train_labeled_dataset, train_unlabeled_dataset, test_dataset