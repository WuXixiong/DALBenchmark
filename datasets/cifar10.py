from torchvision import datasets
from torch.utils.data import Dataset, Subset
import numpy as np
from arguments import parser

import random
import torch

args = parser.parse_args()
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.backends.cudnn.deterministic = True

class MyCIFAR10(Dataset):
    def __init__(self, file_path, train, download, transform):
        self.cifar10 = datasets.CIFAR10(file_path, train=train, download=download, transform=transform)
        self.targets = np.array(self.cifar10.targets)
        self.classes = self.cifar10.classes

        args = parser.parse_args()
        if args.method == 'TIDAL':
            self.moving_prob = np.zeros((len(self.cifar10), int(args.n_class)), dtype=np.float32)
        
        if args.imbalanceset:
            # Create imbalance ratios
            imbalance_ratios = np.logspace(np.log10(args.imb_factor), 0, num=10)[::-1]

            # Get index of each class
            train_targets = np.array(self.cifar10.targets)
            train_idx_per_class = [np.where(train_targets == i)[0] for i in range(10)]

            # Resample according to the indices
            new_indices = []
            for class_idx, class_indices in enumerate(train_idx_per_class):
                n_samples = int(len(class_indices) * imbalance_ratios[class_idx])
                new_indices.extend(np.random.choice(class_indices, n_samples, replace=False))

            # Create the imbalanced train dataset
            self.cifar10.data = self.cifar10.data[new_indices]
            self.targets = self.targets[new_indices]

    def __getitem__(self, index):
        args = parser.parse_args()
        if args.method == 'TIDAL':
            data, _ = self.cifar10[index]
            target = self.targets[index]
            moving_prob = self.moving_prob[index]
            return data, target, index, moving_prob

        data, _ = self.cifar10[index]
        target = self.targets[index]
        return data, target, index

    def __len__(self):
        return len(self.cifar10)