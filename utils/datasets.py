import pickle

import pandas as pd
import torch
import torch.nn as nn
import torchvision.transforms as T
from torchvision.datasets import MNIST, CIFAR10
from torch.utils import data
from torch.utils.data import random_split
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset

from datasets.celeba import CelebA1000, CelebAAttr
from datasets.facescrub import FaceScrub
from datasets.stanford_dogs import StanfordDogs


def get_normalization():
    normalization = T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    return normalization


def get_train_val_split(data, split_ratio, seed=0):
    validation_set_length = int(split_ratio * len(data))
    training_set_length = len(data) - validation_set_length
    torch.manual_seed(seed)
    training_set, validation_set = random_split(
        data, [training_set_length, validation_set_length])

    return training_set, validation_set


def get_subsampled_dataset(dataset,
                           dataset_size=None,
                           proportion=None,
                           seed=0):
    if dataset_size > len(dataset):
        raise ValueError(
            'Dataset size is smaller than specified subsample size')
    if dataset_size is None:
        if proportion is None:
            raise ValueError('Neither dataset_size nor proportion specified')
        else:
            dataset_size = int(proportion * len(dataset))
    torch.manual_seed(seed)
    subsample, _ = random_split(
        dataset, [dataset_size, len(dataset) - dataset_size])
    return subsample


def get_facescrub_idx_to_class():
    with open('utils/files/facescrub_idx_to_class.pkl', 'rb') as f:
        idx_to_class = pickle.load(f)
    return idx_to_class


def get_facescrub_class_to_idx():
    with open('utils/files/facescrub_class_to_idx.pkl', 'rb') as f:
        class_to_idx = pickle.load(f)
    return class_to_idx


def get_celeba_idx_to_attr(list_attr_file='data/celeba/list_attr_celeba.txt'):
    file = pd.read_csv(list_attr_file)
    attributes = file.iloc[0].tolist()[0].split(' ')[:-1]
    attr_dict = {idx: attributes[idx] for idx in range(len(attributes))}
    return attr_dict


def get_celeba_attr_to_idx(list_attr_file='data/celeba/list_attr_celeba.txt'):
    file = pd.read_csv(list_attr_file)
    attributes = file.iloc[0].tolist()[0].split(' ')[:-1]
    attr_dict = {attributes[idx]: idx for idx in range(len(attributes))}
    return attr_dict


def get_stanford_dogs_idx_to_class():
    with open('utils/files/stanford_dogs_idx_to_class.pkl', 'rb') as f:
        idx_to_class = pickle.load(f)
    return idx_to_class


def get_stanford_dogs_class_to_idx():
    with open('utils/files/stanford_dogs_class_to_idx.pkl', 'rb') as f:
        class_to_idx = pickle.load(f)
    return class_to_idx


def create_target_dataset(dataset_config, transform):
    dataset_name = dataset_config['type']
    dataset_path = dataset_config['path']
    if dataset_name.lower() == 'facescrub':
        return FaceScrub(group='all',
                         train=True,
                         transform=transform)
    elif dataset_name.lower() == 'celeba_identities':
        return CelebA1000(train=True, transform=transform, root=dataset_path)
    elif dataset_name.lower() == 'celeba_attr':
        return CelebAAttr(train=True, transform=transform, root=dataset_path)
    elif dataset_name.lower() == 'mnist':
        return MNIST(dataset_path, 
                     train=True, download=True,
                     transform=T.Compose([
                                    T.ToTensor(),
                                    T.Normalize(
                                        (0.1307,), (0.3081,))
                                ]))
    elif dataset_name.lower() == 'cifar10':
        return CIFAR10(dataset_path, 
                     train=True, download=True,
                     transform=transform)
    elif 'stanford_dogs' in dataset_name.lower():
        return StanfordDogs(train=True, cropped=True, transform=transform, root=dataset_path)
    else:
        print(f'{dataset_name} is no valid dataset.')
