
import numpy as np 
import tensorflow_datasets as tfds
from .augmentations import get_transform

DATASETS = ["cifar10", "cifar100"]


def load_dataset_as_numpy(name):
    assert name in DATASETS, ValueError(f"Unrecognized dataset {name}, expected one of {DATASETS}")
    train_ds, test_ds = tfds.load(name, split=["train", "test"], as_supervised=True, batch_size=-1)
    train_imgs, train_labels = tfds.as_numpy(train_ds)
    test_imgs, test_labels = tfds.as_numpy(test_ds)
    train_imgs = train_imgs.astype(np.float32) / 255.0
    test_imgs = test_imgs.astype(np.float32) / 255.0
    return (train_imgs, train_labels), (test_imgs, test_labels)


# ================================================================================
# DATASET CLASSES
# ================================================================================

class FeaturesDataset:

    def __init__(self, features, labels, shuffle=False):
        self.inputs = features
        self.labels = labels 
        self.return_items = ["features", "label"]
        if shuffle:
            order = np.random.permutation(np.arange(len(self.inputs)))
            self.inputs, self.labels = self.inputs[order], self.labels[order]

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        fvec = self.inputs[idx]
        label = self.labels[idx]
        return {"features": fvec, "label": label}


class SimclrDataset:

    def __init__(self, images, labels, transform, shuffle=False):
        self.inputs = images 
        self.labels = labels 
        self.transform = transform
        self.return_items = ["orig", "aug_1", "aug_2", "label"]
        if shuffle:
            order = np.random.permutation(np.arange(len(self.inputs)))
            self.inputs, self.labels = self.inputs[order], self.labels[order]

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        orig_img = self.inputs[idx]
        label = self.labels[idx]
        aug1 = self.transform(orig_img)
        aug2 = self.transform(orig_img)
        return {"orig": orig_img, "aug_1": aug1, "aug_2": aug2, "label": label}


# =======================================================================================
# DATALOADER CLASS
# =======================================================================================

class DataLoader:

    def __init__(self, dataset, batch_size):
        self.ptr = 0
        self.dataset = dataset
        self.batch_size = batch_size

    def __len__(self):
        return len(self.dataset) // self.batch_size
    
    def get(self):
        return_set = {k: [] for k in self.dataset.return_items}
        for _ in range(self.batch_size):
            for k, v in self.dataset[self.ptr].items():
                return_set[k].append(v)
                self.ptr += 1
            
            if self.ptr == len(self.dataset):
                self.ptr = 0 
                break

        return_set = {k: np.stack(v, axis=0) for k, v in return_set.items()}
        return return_set


# ===================================================================================================
# DATALOADER HELPERS
# ===================================================================================================

def get_simclr_dataloaders(dataset_name, transforms, batch_size):
    (train_imgs, train_labels), (test_imgs, test_labels) = load_dataset_as_numpy(dataset_name)
    train_dset = SimclrDataset(images=train_imgs, labels=train_labels, transform=get_transform(transforms["train"]), shuffle=True)
    test_dset = SimclrDataset(images=test_imgs, labels=test_labels, transform=get_transform(transforms["test"]), shuffle=False)
    train_loader = DataLoader(train_dset, batch_size=batch_size)
    test_loader = DataLoader(test_dset, batch_size=batch_size)
    return train_loader, test_loader