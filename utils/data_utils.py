
import torch
import numpy as np
from torchvision import datasets 
from torch.utils.data import Dataset, DataLoader
from .augmentations import get_transform

DATASETS = {
    "cifar10": datasets.CIFAR10,
    "cifar100": datasets.CIFAR100
}


# ================================================================================
# DATASET CLASSES
# ================================================================================

class FeaturesDataset:

    def __init__(self, features, labels):
        self.inputs = features
        self.labels = labels 
        self.return_items = ["features", "label"]

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        fvec = self.inputs[idx]
        label = self.labels[idx]
        return {"features": fvec, "label": label}


class SimclrDataset(Dataset):

    def __init__(self, dataset, transforms):
        super(SimclrDataset, self).__init__()
        self.dataset = dataset 
        self.train_transform = get_transform(transforms["train"])
        self.test_transform = get_transform(transforms["test"])
        self.return_items = ["img", "aug_1", "aug_2", "label"]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        orig_img, label = self.dataset[idx]
        img = self.test_transform(orig_img)
        aug1 = self.train_transform(orig_img)
        aug2 = self.train_transform(orig_img)
        return {"img": img, "aug_1": aug1, "aug_2": aug2, "label": label}


# ===================================================================================================
# DATALOADER HELPERS
# ===================================================================================================

def get_feature_dataloaders(features, labels, batch_size):
    train_dset = FeaturesDataset(features=features, labels=labels)
    test_dset = FeaturesDataset(features=features, labels=labels)
    train_loader = DataLoader(train_dset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader

def get_simclr_dataloaders(dataset_name, root, transforms, batch_size):
    assert dataset_name in DATASETS.keys(), f"Unrecognized dataset {dataset_name}, expected one of {list(DATASETS.keys())}"
    train_dset = DATASETS[dataset_name](root=root, train=True, transform=None, download=True)
    test_dset = DATASETS[dataset_name](root=root, train=False, transform=None, download=True)
    simclr_train_dset = SimclrDataset(dataset=train_dset, transforms=transforms)
    simclr_test_dset = SimclrDataset(dataset=test_dset, transforms=transforms)
    train_loader = DataLoader(simclr_train_dset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(simclr_test_dset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader