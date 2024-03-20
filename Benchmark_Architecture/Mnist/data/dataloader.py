import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader

def get_data_loaders(root="../../data/", batch_size=128, num_workers=2):
    """
    MNIST 데이터셋에 대한 데이터로더를 반환합니다.

    Args:
        batch_size (int): 배치 크기입니다. 기본값은 128입니다.
        num_workers (int): 데이터를 로드하는 데 사용되는 스레드 수입니다. 기본값은 2입니다.

    Returns:
        train_loader (DataLoader): 학습 데이터셋에 대한 데이터 로더입니다.
        test_loader (DataLoader): 테스트 데이터셋에 대한 데이터 로더입니다.
    """
    mnist_transform = transforms.Compose([transforms.ToTensor()])

    train_dataset = datasets.MNIST(root=root, train=True, download=True, transform=mnist_transform)
    test_dataset = datasets.MNIST(root=root, train=False, download=True, transform=mnist_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    return train_loader, test_loader
