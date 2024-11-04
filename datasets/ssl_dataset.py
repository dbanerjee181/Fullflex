import torch
import torchvision
from torchvision import transforms as T  # Import PyTorch transforms

from .data_utils import split_ssl_data  # Assuming this is adapted for PyTorch
from .dataset import BasicDataset  # Assuming this is adapted for PyTorch
import numpy as np
import json
import os
import random
import sys
import copy

def get_transform(crop_size, train=True):
    if train:
        transform = T.Compose([
            T.RandomHorizontalFlip(),  # Random horizontal flip
            T.RandomCrop(crop_size, padding=4)  # Random crop with padding
        ])
        return transform
    return None

class SSL_Dataset:
    def __init__(self, args, name='cifar10', train=True, num_classes=10, data_dir='./data'):
        self.args = args
        self.name = name
        self.train = train
        self.num_classes = num_classes
        self.data_dir = data_dir
        
        # Determine crop size based on dataset name
        crop_size = 96 if self.name.upper() == 'STL10' else 224 if self.name.upper() == 'IMAGENET' else 32
        
        # Get weak augmentation transform
        self.transform = get_transform(crop_size, train)  # Assuming get_transform is adapted for PyTorch
        
    def get_data(self, svhn_extra=True):
            if self.name.upper() == 'SVHN':
              # Load SVHN dataset using torchvision.datasets
                if self.train:
                    dataset = torchvision.datasets.SVHN(
                    root=self.data_dir, split='train', download=True, transform=None
                    )
                    if svhn_extra:
                        extra_dataset = torchvision.datasets.SVHN(
                        root=self.data_dir, split='extra', download=True, transform=None
                        )
                        dataset.data = np.concatenate([dataset.data, extra_dataset.data])
                        dataset.labels = np.concatenate([dataset.labels, extra_dataset.labels])
                else:
                    dataset = torchvision.datasets.SVHN(
                    root=self.data_dir, split='test', download=True, transform=None
                    )
                data = dataset.data  # Get data (images)
                targets = dataset.labels  # Get targets (labels)
        
            elif self.name.upper() == 'STL10':
                # Load STL10 dataset using torchvision.datasets
                dataset = torchvision.datasets.STL10(
                          root=self.data_dir, split='train' if self.train else 'test', 
                          download=True, transform=None
                        )
                data = dataset.data  # Get data (images)
                targets = dataset.labels  # Get targets (labels)

            else:
                # Load other datasets (e.g., CIFAR10) using torchvision.datasets
                dataset_class = getattr(torchvision.datasets, self.name.upper())
                dataset = dataset_class(
                    root=self.data_dir, train=self.train, download=True, transform=None
                    )
                data = dataset.data  # Get data (images)
                targets = dataset.targets  # Get targets (labels)

            # Convert data to RGB format if necessary
            if data.shape[3] == 3:  # Check if data is in BGR format
                data = data[:, :, :, ::-1]  # Convert BGR to RGB
        
            return data, targets
    def get_dset(self, is_ulb=False, strong_transform=None):
        data, targets = self.get_data()  # Assuming get_data is adapted for PyTorch
        num_classes = self.num_classes
        transform = self.transform  # Assuming self.transform is a PyTorch transform

        # Create and return a BasicDataset instance (adapted for PyTorch)
        return BasicDataset(self.name, data, targets, num_classes, transform, is_ulb, strong_transform)

    def get_ssl_dset(self, num_labels, index=None, include_lb_to_ulb=True, strong_transform=None):
        data, targets = self.get_data()  # Assuming get_data is adapted for PyTorch

        # Split data into labeled and unlabeled sets (assuming split_ssl_data is adapted for PyTorch)
        lb_data, lb_targets, ulb_data, ulb_targets = split_ssl_data(
            self.args, data, targets, num_labels, self.num_classes, index, include_lb_to_ulb
        )

        # Create and return BasicDataset instances for labeled and unlabeled data
        lb_dset = BasicDataset(self.name, lb_data, lb_targets, self.num_classes, self.transform, False, None)
        ulb_dset = BasicDataset(self.name, ulb_data, ulb_targets, self.num_classes, self.transform, True, strong_transform)
        return lb_dset, ulb_dset