
import os
import jax
import numpy as np

from torchvision import datasets
from .transforms import build_transform


def split_across_devices(xs):
    return jax.tree_util.tree_map(
        lambda x: x.reshape((jax.local_device_count(), -1) + x.shape[1:]) \
            if len(x.shape) != 0 else x, 
        xs
    )


def numpy_collate(batch):
    if isinstance(batch[0], np.ndarray):
        batch = np.stack(batch)
    elif isinstance(batch[0], (tuple, list)):
        transposed = zip(*batch)
        batch = [numpy_collate(samples) for samples in transposed]
    else:
        batch = np.array(batch)
    batch = split_across_devices(batch)
    print(batch[0].shape, batch[1].shape)
    exit()
    return batch
    
    
def get_datasets(name, root, train_transform_cfg, val_transform_cfg):
    assert name in ['cifar10', 'cifar100', 'imagenet1k'], \
        f'name should be one of ["cifar10", "cifar100", "imagenet1k"]'
        
    train_transform = build_transform(train_transform_cfg)
    val_transform = build_transform(val_transform_cfg)
    
    if name == 'cifar10':
        trainset = datasets.CIFAR10(root=root, train=True, transform=train_transform, download=True)
        valset = datasets.CIFAR10(root=root, train=False, transform=val_transform, download=True)
    
    elif name == 'cifar100':
        trainset = datasets.CIFAR100(root=root, train=True, transform=train_transform, download=True)
        valset = datasets.CIFAR100(root=root, train=False, transform=val_transform, download=True)
    
    elif name == 'imagenet1k':
        trainset = datasets.ImageFolder(root=os.path.join(root, 'train'), transform=train_transform)
        valset = datasets.ImageFolder(root=os.path.join(root, 'val'), transform=val_transform)
        
    return trainset, valset