# from torchvision import datasets
# from torch.utils.data import Dataset, Subset
# import numpy as np
# from arguments import parser

# import random
# import torch

# args = parser.parse_args()
# random.seed(args.seed)
# np.random.seed(args.seed)
# torch.manual_seed(args.seed)
# torch.backends.cudnn.deterministic = True

# class MyCIFAR10(Dataset):
#     def __init__(self, file_path, train, download, transform):
#         self.cifar10 = datasets.CIFAR10(file_path, train=train, download=download, transform=transform)
#         self.targets = np.array(self.cifar10.targets)
#         self.classes = self.cifar10.classes

#         args = parser.parse_args()
#         if args.method == 'TIDAL':
#             self.moving_prob = np.zeros((len(self.cifar10), int(args.n_class)), dtype=np.float32)
        
#         if args.imbalanceset:
#             # Create imbalance ratios
#             imbalance_ratios = np.logspace(np.log10(args.imb_factor), 0, num=10)[::-1]

#             # Get index of each class
#             train_targets = np.array(self.cifar10.targets)
#             train_idx_per_class = [np.where(train_targets == i)[0] for i in range(10)]

#             # Resample according to the indices
#             new_indices = []
#             for class_idx, class_indices in enumerate(train_idx_per_class):
#                 n_samples = int(len(class_indices) * imbalance_ratios[class_idx])
#                 new_indices.extend(np.random.choice(class_indices, n_samples, replace=False))

#             # Create the imbalanced train dataset
#             self.cifar10.data = self.cifar10.data[new_indices]
#             self.targets = self.targets[new_indices]

#     def __getitem__(self, index):
#         args = parser.parse_args()
#         if args.method == 'TIDAL':
#             data, _ = self.cifar10[index]
#             target = self.targets[index]
#             moving_prob = self.moving_prob[index]
#             return data, target, index, moving_prob

#         data, _ = self.cifar10[index]
#         target = self.targets[index]
#         return data, target, index

#     def __len__(self):
#         return len(self.cifar10)

from torch.utils.data import Dataset
import numpy as np
import torch

class MyCIFAR10(Dataset):
    def __init__(self, hf_dataset, transform=None, imbalance_factor=None, method=None, n_class=10):
        """
        初始化 CIFAR-10 数据集。

        参数：
        hf_dataset (Dataset): Hugging Face 加载的数据集对象。
        transform (callable, optional): 对图像进行预处理的函数。
        imbalance_factor (float, optional): 数据集不平衡因子。
        method (str, optional): 使用的方法名称。
        n_class (int): 类别数量，默认为 10。
        """
        self.data = hf_dataset['img']
        self.targets = np.array(hf_dataset['label'])
        self.transform = transform
        self.classes = hf_dataset.features['label'].names

        if method == 'TIDAL':
            self.moving_prob = np.zeros((len(self.data), n_class), dtype=np.float32)

        if imbalance_factor:
            # 创建不平衡比例
            imbalance_ratios = np.logspace(np.log10(imbalance_factor), 0, num=n_class)[::-1]

            # 获取每个类别的索引
            train_idx_per_class = [np.where(self.targets == i)[0] for i in range(n_class)]

            # 根据索引重新采样
            new_indices = []
            for class_idx, class_indices in enumerate(train_idx_per_class):
                n_samples = int(len(class_indices) * imbalance_ratios[class_idx])
                new_indices.extend(np.random.choice(class_indices, n_samples, replace=False))

            # 创建不平衡的训练数据集
            self.data = [self.data[i] for i in new_indices]
            self.targets = self.targets[new_indices]

    def __getitem__(self, index):
        """
        根据索引获取样本。

        参数：
        index (int): 样本的索引。

        返回：
        tuple: (图像, 标签, 索引, [移动概率])
        """
        img = self.data[index]
        target = self.targets[index]

        if self.transform:
            img = self.transform(img)

        if hasattr(self, 'moving_prob'):
            moving_prob = self.moving_prob[index]
            return img, target, index, moving_prob

        return img, target, index

    def __len__(self):
        """
        返回数据集的样本数量。

        返回：
        int: 样本数量。
        """
        return len(self.data)
