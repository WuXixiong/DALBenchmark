from torchvision import datasets
from torch.utils.data import Dataset
import numpy as np
from arguments import parser  # Assuming parser is defined in the arguments module

class MySVHN(Dataset):
    def __init__(self, file_path, split, download, transform):
        self.svhn = datasets.SVHN(file_path, split=split, download=download, transform=transform)
        self.targets = np.array(self.svhn.labels)  # SVHN labels are stored in `labels`
        self.classes = list(range(10))  # SVHN has 10 classes: digits 0-9

        args = parser.parse_args()
        if args.method == 'TIDAL':
            self.moving_prob = np.zeros((len(self.svhn), int(args.n_class)), dtype=np.float32)

        args = parser.parse_args()
        
        if args.imbalanceset:
            # Create imbalance ratios
            imbalance_ratios = np.logspace(np.log10(args.imb_factor), 0, num=10)[::-1]

            # Get index of each class
            train_targets = np.array(self.svhn.targets)
            train_idx_per_class = [np.where(train_targets == i)[0] for i in range(10)]

            # Resample according to the indices
            new_indices = []
            for class_idx, class_indices in enumerate(train_idx_per_class):
                n_samples = int(len(class_indices) * imbalance_ratios[class_idx])
                new_indices.extend(np.random.choice(class_indices, n_samples, replace=False))

            # Create the imbalanced train dataset
            self.svhn.data = self.svhn.data[new_indices]
            self.targets = self.targets[new_indices]

    def __getitem__(self, index):
        args = parser.parse_args()
        if args.method == 'TIDAL':
            data, _ = self.svhn[index]
            target = self.targets[index]
            moving_prob = self.moving_prob[index]
            return data, target, index, moving_prob
        
        data, target = self.svhn[index]  # SVHN already returns (data, target)
        return data, target, index

    def __len__(self):
        return len(self.svhn)
