import math
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import DataLoader, RandomSampler
from model import WideResNet

def set_model():
    # model = models.wide_resnet50_2(pretrained=False)
    # model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    # model.maxpool = nn.Identity()
    # model.fc = nn.Linear(model.fc.in_features, 10)
    model = WideResNet(depth = 28,
                    num_classes = 10,
                    widen_factor = 2,
                    bn_momentum = 0.01,
                    leaky_slope = 0.1,
                    dropRate = 0.0)
    return model

def initial_model():
    train_model = set_model()
    eval_model = set_model()
    for param_q, param_k in zip(train_model.parameters(), eval_model.parameters()):
            param_k.data.copy_(param_q.detach().data)  
            param_k.requires_grad = False
    return train_model, eval_model

def get_cosine_schedule_with_warmup(optimizer,
                                    num_training_steps,
                                    num_cycles=7. / 16.,
                                    num_warmup_steps=0,
                                    last_epoch=-1):
    from torch.optim.lr_scheduler import LambdaLR
    def _lr_lambda(current_step):
        if current_step < num_warmup_steps:
            _lr = float(current_step) / float(max(1, num_warmup_steps))
        else:
            num_cos_steps = float(current_step - num_warmup_steps)
            num_cos_steps = num_cos_steps / float(max(1, num_training_steps - num_warmup_steps))
            _lr = max(0.0, math.cos(math.pi * num_cycles * num_cos_steps))
        return _lr

    return LambdaLR(optimizer, _lr_lambda, last_epoch)

def get_data_loader(dataset, batch_size, num_iters, num_workers=2, pin_memory=True, drop_last=True):
    num_samples = batch_size * num_iters
    data_sampler = RandomSampler(dataset, replacement=True, num_samples=num_samples)
    data_loader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        sampler=data_sampler,
        num_workers=num_workers, 
        pin_memory=pin_memory, 
        drop_last=drop_last
    )
    
    return data_loader