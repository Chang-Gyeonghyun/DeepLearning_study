import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import Dataset, Sampler
from torchvision.datasets import ImageFolder
from collections import defaultdict
import random

dataroot_source = "../s3d/data/OfficeHomeDataset_10072016/Real World"
dataroot_target = "../s3d/data/OfficeHomeDataset_10072016/Real World"

train_transform = transforms.Compose([
    transforms.Resize((64,64)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

test_transform = transforms.Compose([
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

def create_custom_imagefolder(dataset, indices):
    paths = [dataset.samples[i][0] for i in indices]
    labels = [dataset.samples[i][1] for i in indices]
    samples = [(paths[i], labels[i]) for i in range(len(paths))]
    new_dataset = ImageFolder(dataset.root, transform=dataset.transform)
    new_dataset.samples = samples
    new_dataset.imgs = samples
    return new_dataset

def get_dataset():
    source_dataset = datasets.ImageFolder(root=dataroot_source,
                                          transform=train_transform)
    target_dataset = datasets.ImageFolder(root=dataroot_target)

    # 전체 dataset 크기 계산
    dataset_size = len(target_dataset)
    train_size = dataset_size
    val_size = int(0.2 * dataset_size)
    test_size = dataset_size - train_size - val_size

    # 데이터셋 분할
    train_indices, val_test_indices = torch.utils.data.random_split(
        range(dataset_size), [train_size, val_size + test_size])
    val_indices, test_indices = torch.utils.data.random_split(
        val_test_indices, [val_size, test_size])

    # 각각의 데이터셋에 해당하는 transform 적용
    test_dataset = datasets.ImageFolder(root=dataroot_target, transform=test_transform)
    test_dataset.samples = [target_dataset.samples[i] for i in test_indices]

    val_dataset = datasets.ImageFolder(root=dataroot_target, transform=test_transform)
    val_dataset.samples = [target_dataset.samples[i] for i in val_indices]

    target_dataset = datasets.ImageFolder(root=dataroot_target, transform=train_transform)
    target_dataset.samples = [target_dataset.samples[i] for i in train_indices]

    # 클래스별 인덱스 정리
    class_indices = defaultdict(list)
    for idx, (_, label) in enumerate(target_dataset.samples):
        class_indices[label].append(idx)

    labeled_indices = []
    unlabeled_indices = []

    # 클래스당 라벨링된 데이터와 비라벨링된 데이터 분할 (클래스당 5개 라벨링된 데이터 사용)
    for label, indices in class_indices.items():
        random.shuffle(indices)
        labeled_indices.extend(indices[:3])
        unlabeled_indices.extend(indices[3:])

    # 커스텀 데이터셋 생성
    labeled_target_dataset = create_custom_imagefolder(target_dataset, labeled_indices)
    unlabeled_target_dataset = create_custom_imagefolder(target_dataset, unlabeled_indices)

    return source_dataset, labeled_target_dataset, unlabeled_target_dataset, val_dataset, test_dataset


class DatasetWrapper(Dataset):
    def __init__(self, source_dataset, labeled_target_dataset, unlabeled_target_dataset):
        self.source_dataset = source_dataset
        self.labeled_target_dataset = labeled_target_dataset
        self.unlabeled_target_dataset = unlabeled_target_dataset

        self.source_len = len(source_dataset)
        self.labeled_target_len = len(labeled_target_dataset)
        self.unlabeled_target_len = len(unlabeled_target_dataset)
    
        self.total_len = self.source_len + self.labeled_target_len + self.unlabeled_target_len
        
        self.classwise_indices = defaultdict(list)
        self._init_classwise_indices()

    def _init_classwise_indices(self):
        for i in range(self.source_len):
            y = self.source_dataset[i][1]
            self.classwise_indices[y].append(i)

        for i in range(self.labeled_target_len):
            y = self.labeled_target_dataset[i][1]
            self.classwise_indices[y].append(i + self.source_len)

        for i in range(self.unlabeled_target_len):
            y = self.unlabeled_target_dataset.samples[i][1]
            self.classwise_indices[y].append(i + self.source_len + self.labeled_target_len)

    def get_class(self, i):
        if i < self.source_len:
            return self.source_dataset[i][1]
        elif i < self.source_len + self.labeled_target_len:
            return self.labeled_target_dataset[i - self.source_len][1]
        else:
            return self.unlabeled_target_dataset.samples[i - self.source_len - self.labeled_target_len][1]

    def __getitem__(self, index):
        if index < self.source_len:
            return self.source_dataset[index]
        elif index < self.source_len + self.labeled_target_len:
            return self.labeled_target_dataset[index - self.source_len]
        else:
            return self.unlabeled_target_dataset.samples[index - self.source_len - self.labeled_target_len]

    def __len__(self):
        return self.total_len



class PairBatchSampler(Sampler):
    def __init__(self, dataset, batch_size, num_iterations=None):
        self.dataset = dataset
        self.in_batch_size = batch_size // 2  # 64 for label batch
        self.in_in_batch_size = self.in_batch_size // 2  # 32 for source batch
        self.num_iterations = num_iterations
        self.source_len = self.dataset.source_len
        self.target_len = self.dataset.labeled_target_len
        self.label_len = self.source_len + self.target_len
        self.total_len = self.dataset.total_len

    def __iter__(self):
        indices = list(range(len(self.dataset)))
        source_indices = indices[:self.source_len]
        target_label_indices = indices[self.source_len:self.source_len + self.target_len]

        random.shuffle(source_indices)
        random.shuffle(target_label_indices)

        offset_s = 0
        offset_t = 0

        for k in range(len(self)):
            batch_indices_s = []
            batch_indices_t = []
            pair_indices = []

            while len(batch_indices_s) <= self.in_in_batch_size:
                if len(batch_indices_s) == self.in_in_batch_size:
                    break
                offset_s = offset_s % self.source_len
                source_index = source_indices[offset_s]
                y = self.dataset.get_class(source_index)
                filter_target = list(filter(lambda x: x >= self.label_len, self.dataset.classwise_indices[y]))

                if len(filter_target) == 0:
                    offset_s = offset_s + 1
                    continue
                else:
                    selected_pair = random.choice(filter_target)

                    batch_indices_s.append(source_index)
                    pair_indices.append(selected_pair)
                    offset_s = offset_s + 1

            while len(batch_indices_t) <= self.in_in_batch_size:
                if len(batch_indices_t) == self.in_in_batch_size:
                    break
                offset_t = offset_t % self.target_len
                target_index = target_label_indices[offset_t]
                y = self.dataset.get_class(target_index)
                filter_target = list(filter(lambda x: x >= self.label_len, self.dataset.classwise_indices[y]))

                if len(filter_target) == 0:
                    offset_t = offset_t + 1
                    continue
                else:
                    selected_pair = random.choice(filter_target)

                    batch_indices_t.append(target_index)
                    pair_indices.append(selected_pair)
                    offset_t = offset_t + 1

            batch_indices = batch_indices_s + batch_indices_t

            # for debugging
            assert (len(batch_indices_s) == self.in_in_batch_size and len(batch_indices_t) == self.in_in_batch_size)
            assert (min(batch_indices) >= 0) and (max(batch_indices) < self.label_len)
            assert (len(pair_indices) == self.in_batch_size)
            assert (min(pair_indices) >= self.label_len) and (max(pair_indices) < self.total_len)

            yield batch_indices + pair_indices

    def __len__(self):
        if self.num_iterations is None:
            return self.source_len // self.in_in_batch_size
        else:
            return self.num_iterations
        

def make_pseudo_label_rss(unlabeled_target_dataset, G, F1, margin, alpha_value=0.7):
    G.eval()
    F1.eval()
    softmax = nn.Softmax(dim=1)
    batch_size = 256
    target_loader_unl = torch.utils.data.DataLoader(unlabeled_target_dataset,
                                                    batch_size=batch_size, num_workers=2,
                                                    shuffle=False, drop_last=False)

    largest_margin = 0
    total_margin = 0
    margin_list = []
    selected_samples = []  
    for batch_idx, (img, label) in enumerate(target_loader_unl): 
        img = img.cuda()
        pseudo_label = torch.zeros(label.shape).cuda()

        with torch.no_grad():
            feature = G(img)
            output = F1(feature)
            pred = softmax(output)

        top_two_class_index = torch.topk(output, 2)[1]
        top_two_class_prob = torch.topk(output, 2)[0]

        max_pred = torch.max(pred, dim=1)[0]

        for j in range(len(pseudo_label)):
            cal_margin = top_two_class_prob[j, 0] - top_two_class_prob[j, 1]
            total_margin += cal_margin
            margin_list.append(float(cal_margin))

            if cal_margin > largest_margin:
                largest_margin = cal_margin

            if cal_margin > margin:
                pseudo_label[j] = top_two_class_index[j, 0]
            elif max_pred[j] > alpha_value:
                pseudo_label[j] = top_two_class_index[j, 0]
            else:
                pseudo_label[j] = -1

            if pseudo_label[j] != -1:
                global_index = batch_idx * batch_size + j
                img_sample, _ = unlabeled_target_dataset[global_index]  
                selected_samples.append((img_sample, int(pseudo_label[j].item())))

    mean_margin = total_margin / len(unlabeled_target_dataset)
    G.train()
    F1.train()
    pseudo_labeled_dataset = create_imagefolder_from_subset(unlabeled_target_dataset, selected_samples)
    return pseudo_labeled_dataset, largest_margin, mean_margin, margin_list

def create_imagefolder_from_subset(original_dataset, selected_samples):
    pseudo_labeled_dataset = ImageFolder(root=original_dataset.root, transform=train_transform)
    pseudo_labeled_dataset.samples = selected_samples
    pseudo_labeled_dataset.targets = [label for _, label in selected_samples]
    return pseudo_labeled_dataset

def return_stage2_dataset(args, G, F1, stage2_margin):
    source_dataset, labeled_target_dataset, unlabeled_target_dataset, _, _ = get_dataset()
    pseudo_labeled_dataset, largest_margin, mean_margin, margin_list = \
        make_pseudo_label_rss(unlabeled_target_dataset, G, F1, stage2_margin, args['alpha_value'])
    
    train_dataset = DatasetWrapper(source_dataset, labeled_target_dataset, pseudo_labeled_dataset)
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_sampler=PairBatchSampler(train_dataset,
                                                              2 * 96, num_iterations=args['pseudo_interval']),num_workers=2)


    return train_loader