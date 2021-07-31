
import numpy as np 
import tensorflow_datasets as tfds

DATASETS = ["cifar10", "cifar100"]


class DataLoader:

    def __init__(self, inputs, labels, batch_size, shuffle=False):
        self.ptr = 0
        self.inputs = inputs
        self.labels = labels 
        self.batch_size = batch_size
        if shuffle:
            order = np.random.permutation(np.arange(self.inputs.shape[0]))
            self.inputs, self.labels = self.inputs[order], self.labels[order]

    def __len__(self):
        return self.inputs.shape[0]
    
    def get(self):
        x = self.inputs[self.ptr: self.ptr+self.batch_size]
        y = self.labels[self.ptr: self.ptr+self.batch_size]
        self.ptr += self.batch_size
        if self.ptr > len(self.inputs):
            self.ptr = 0


def load_dataset_as_numpy(name):
    assert name in DATASETS, ValueError(f"Unrecognized dataset {name}, expected one of {DATASETS}")
    train_ds, test_ds = tfds.load(name, split=["train", "test"], as_supervised=True, batch_size=-1)
    train_imgs, train_labels = tfds.as_numpy(train_ds)
    test_imgs, test_labels = tfds.as_numpy(test_ds)
    train_imgs = train_imgs.astype(np.float32) / 255.0
    test_imgs = test_imgs.astype(np.float32) / 255.0
    return (train_imgs, train_labels), (test_imgs, test_labels)

def get_dataloaders(name, batch_size):
    (train_imgs, train_labels), (test_imgs, test_labels) = load_dataset_as_numpy(name)
    train_loader = DataLoader(train_imgs, train_labels, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_imgs, test_labels, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader