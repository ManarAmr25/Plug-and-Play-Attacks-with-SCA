import torch
import numpy as np
from .celeba import CelebAAttr

class Subset(torch.utils.data.Dataset):
    def __init__(self, dataset, indices, transform):
        self.dataset = dataset
        self.indices = indices
        self.transform = transform
        self.targets = np.array(dataset.targets)[self.indices]

    def __getitem__(self, idx):
        im, targets = self.dataset[self.indices[idx]]
        if self.transform:
            im = self.transform(im)
        return im, targets

    def __len__(self):
        return len(self.indices)


class SingleClassSubset(torch.utils.data.Dataset):
    def __init__(self, dataset, target_class):
        if isinstance(dataset, CelebAAttr):
            # print("Condition happened !")
            current_targets  = [abs(arr.sum())-1 for arr in dataset.targets]
        else:
            current_targets = dataset.targets
        self.dataset = dataset
        # print("@@@@@@@@@@@@@@@@@@")
        # print(current_targets[:5])
        # print(len(current_targets))
        # # print(len(current_targets[0]))
        # print("@@@@@@@@@@@@@@@@@@")
        self.indices = np.where(np.array(current_targets) == target_class)[0][:5000]
        self.targets = np.array(current_targets)[self.indices]
        self.target_class = target_class
        # if len(self.indices)>1:
            # print("-----------------?",self.dataset[self.indices[0]])


    def __getitem__(self, idx):
        im, targets = self.dataset[self.indices[idx]]
        # print("in single class subset",targets)
        return im, targets

    def __len__(self):
        return len(self.indices)


class ClassSubset(torch.utils.data.Dataset):
    def __init__(self, dataset, target_classes):
        self.dataset = dataset
        if isinstance(dataset, CelebAAttr):
            # print("Condition happened !")
            current_targets  = [abs(arr.sum())-1 for arr in dataset.targets]
        else:
            current_targets = dataset.targets
        self.indices = np.where(
            np.isin(np.array(current_targets), np.array(target_classes)))[0][:5000]
        self.targets = np.array(current_targets)[self.indices]
        self.target_classes = target_classes

    def __getitem__(self, idx):
        im, targets = self.dataset[self.indices[idx]]
        return im, targets

    def __len__(self):
        return len(self.indices)
