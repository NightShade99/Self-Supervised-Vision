
import torch
import numpy as np
from torchvision import datasets 
from torch.utils.data import Dataset, DataLoader
from .augmentations import get_transform, MultiCrop

DATASETS = {
    "cifar10": datasets.CIFAR10,
    "cifar100": datasets.CIFAR100
}


# ================================================================================
# DATASET CLASSES
# ================================================================================

class FeaturesDataset(Dataset):

    def __init__(self, features, labels):
        super(FeaturesDataset, self).__init__()
        self.inputs = features
        self.labels = labels 
        self.return_items = ["features", "label"]

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        fvec = self.inputs[idx]
        label = self.labels[idx]
        return {"features": fvec, "label": label}


class PseudoLabelDataset(Dataset):

    def __init__(self, dataset, transforms, labels=None):
        super(PseudoLabelDataset, self).__init__()
        self.dataset = dataset
        self.labels = labels
        self.aug_transform = get_transform(transforms["aug"])
        self.std_transform = get_transform(transforms["std"])
        self.return_items = ["img", "aug", "label"]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        orig_img, gt = self.dataset[idx]
        aug = self.aug_transform(orig_img)
        img = self.std_transform(orig_img)
        label = self.labels[idx] if self.labels is not None else gt 
        return {"idx": idx, "img": img, "aug": aug, "label": label}


class DoubleAugmentedDataset(Dataset):

    def __init__(self, dataset, transforms):
        super(DoubleAugmentedDataset, self).__init__()
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
        return {"index": idx, "img": img, "aug_1": aug1, "aug_2": aug2, "label": label}


class MultiCropDataset(Dataset):

    def __init__(self, dataset, multicrop_config):
        super(MultiCropDataset, self).__init__()
        self.dataset = dataset
        self.multi_crop = MultiCrop(multicrop_config)
        self.test_transform = get_transform(multicrop_config["test_transforms"])
        self.return_items = ["img", "global", "local", "label"]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        orig_img, label = self.dataset[idx]
        img = self.test_transform(orig_img)
        global_1, global_2, local_1, local_2 = self.multi_crop(orig_img).values()
        return {"img": img, "global_1": global_1, "global_2": global_2, "local_1": local_1, "local_2": local_2, "label": label}


# ===================================================================================================
# DATALOADER HELPERS
# ===================================================================================================

def get_feature_dataloaders(features, labels, batch_size):
    train_dset = FeaturesDataset(features=features, labels=labels)
    test_dset = FeaturesDataset(features=features, labels=labels)
    train_loader = DataLoader(train_dset, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dset, batch_size=batch_size, shuffle=False, num_workers=4)
    return train_loader, test_loader

def get_pseudolabel_dataloaders(train_dset, test_dset, train_labels, root, transforms, batch_size):
    train_dataset = PseudoLabelDataset(dataset=train_dset, transforms=transforms, labels=train_labels)
    test_dataset = PseudoLabelDataset(dataset=test_dset, transforms=transforms, labels=None)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    return train_loader, test_loader

def get_double_augment_dataloaders(dataset_name, root, transforms, batch_size):
    assert dataset_name in DATASETS.keys(), f"Unrecognized dataset {dataset_name}, expected one of {list(DATASETS.keys())}"
    train_dset = DATASETS[dataset_name](root=root, train=True, transform=None, download=True)
    test_dset = DATASETS[dataset_name](root=root, train=False, transform=None, download=True)
    train_dset = DoubleAugmentedDataset(dataset=train_dset, transforms=transforms)
    test_dset = DoubleAugmentedDataset(dataset=test_dset, transforms=transforms)
    train_loader = DataLoader(train_dset, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dset, batch_size=batch_size, shuffle=False, num_workers=4)
    return train_loader, test_loader

def get_multicrop_dataloaders(dataset_name, root, multicrop_config, batch_size):
    assert dataset_name in DATASETS.keys(), f"Unrecognized dataset {dataset_name}, expected one of {list(DATASETS.keys())}"
    train_dset = DATASETS[dataset_name](root=root, train=True, transform=None, download=True)
    test_dset = DATASETS[dataset_name](root=root, train=False, transform=None, download=True)
    train_dset = MultiCropDataset(dataset=train_dset, multicrop_config=multicrop_config)
    test_dset = MultiCropDataset(dataset=test_dset, multicrop_config=multicrop_config)
    train_loader = DataLoader(train_dset, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dset, batch_size=batch_size, shuffle=False, num_workers=4)
    return train_loader, test_loader