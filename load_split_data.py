import pickle
import numpy as np
import torch
import math
import random
from torch.utils.data.dataset import Subset
from datasets.tinyimagenet import MyTinyImageNet
from datasets.cifar10 import MyCIFAR10
from datasets.cifar100 import MyCIFAR100
from datasets.imagenet import MyImageNet
from datasets.mnist import MyMNIST
from datasets.svhn import MySVHN
from datasets.agnews import MyAGNewsDataset
from datasets.imdb import MyIMDBDataset
from datasets.sst5 import MySST5Dataset
from torchvision import datasets
import torchvision.transforms as T
from transformers import DistilBertTokenizer
from transformers import RobertaTokenizer
import os
from tqdm import tqdm

CIFAR10_SUPERCLASS = list(range(10))  # one class
CIFAR100_SUPERCLASS = [
    [4, 31, 55, 72, 95],#1  d
    [1, 33, 67, 73, 91],#2  d
    [54, 62, 70, 82, 92],#3
    [9, 10, 16, 29, 61],#4
    [0, 51, 53, 57, 83],#5
    [22, 25, 40, 86, 87],#6
    [5, 20, 26, 84, 94],#7
    [6, 7, 14, 18, 24],#8   d
    [3, 42, 43, 88, 97],#9   d
    [12, 17, 38, 68, 76],#10
    [23, 34, 49, 60, 71],#11
    [15, 19, 21, 32, 39],#12  d
    [35, 63, 64, 66, 75],#13  d
    [27, 45, 77, 79, 99],#14
    [2, 11, 36, 46, 98],#15
    [28, 30, 44, 78, 93],#16   d
    [37, 50, 65, 74, 80],#17   d
    [47, 52, 56, 59, 96],#18
    [8, 13, 48, 58, 90],#19
    [41, 69, 81, 85, 89],#20
]
IMAGENET_SUPERCLASS = list(range(30))  # one class

def get_subset_with_len(dataset, length, shuffle=False):
    dataset_size = len(dataset)

    index = np.arange(dataset_size)
    if shuffle:
        np.random.shuffle(index)

    index = torch.from_numpy(index[0:length])
    subset = Subset(dataset, index)

    assert len(subset) == length

    return subset

def get_dataset(args, trial):
    # Normalization for image datasets
    if args.dataset == 'CIFAR10':
        T_normalize = T.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
    elif args.dataset == 'CIFAR100':
        T_normalize = T.Normalize([0.5071, 0.4867, 0.4408], [0.2675, 0.2565, 0.2761])
    elif args.dataset == 'ImageNet50' or args.dataset == 'TinyImageNet':
        T_normalize = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    elif args.dataset == 'MNIST':
        T_normalize = T.Normalize([0.1307], [0.3081])  # Mean and std for MNIST
    elif args.dataset == 'SVHN':
        T_normalize = T.Normalize([0.4377, 0.4438, 0.4728], [0.1980, 0.2010, 0.1970])  # Mean and std for SVHN

    # Transform
    if args.dataset in ['CIFAR10', 'CIFAR100', 'SVHN']:
        train_transform = T.Compose([T.RandomHorizontalFlip(), T.RandomCrop(size=32, padding=4), T.ToTensor(), T_normalize])  #
        test_transform = T.Compose([T.ToTensor(), T_normalize])
    elif args.dataset == 'ImageNet50':
        train_transform = T.Compose([T.Resize(256), T.RandomCrop(224), T.RandomHorizontalFlip(), T.ToTensor(), T_normalize])
        test_transform = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor(), T_normalize])
    elif args.dataset == 'TinyImageNet':
        train_transform = T.Compose([T.Resize(64), T.RandomCrop(64, padding=4), T.RandomHorizontalFlip(), T.ToTensor(), T_normalize])
        test_transform = T.Compose([T.Resize(64), T.ToTensor(), T_normalize])
    elif args.dataset == 'MNIST':
        train_transform = T.Compose([
            T.RandomRotation(10),  # Randomly rotate the image by 10 degrees
            T.RandomCrop(28, padding=4),  # Randomly crop with padding
            T.ToTensor(),
            T_normalize  # Normalize based on mean and std for MNIST
        ])
        test_transform = T.Compose([
            T.ToTensor(),
            T_normalize
        ])

    # Dataset loading
    if args.dataset == 'CIFAR10':
        file_path = args.data_path + '/cifar10/'
        train_set = MyCIFAR10(file_path, train=True, download=True, transform=train_transform)
        unlabeled_set = MyCIFAR10(file_path, train=True, download=True, transform=test_transform)
        test_set = MyCIFAR10(file_path, train=False, download=True, transform=test_transform)
    elif args.dataset == 'CIFAR100':
        file_path = args.data_path + '/cifar100/'
        train_set = MyCIFAR100(file_path, train=True, download=True, transform=train_transform)
        unlabeled_set = MyCIFAR100(file_path, train=True, download=True, transform=test_transform)
        test_set = MyCIFAR100(file_path, train=False, download=True, transform=test_transform)
    elif args.dataset == 'MNIST':
        file_path = args.data_path + '/mnist/'  # Change the folder to store MNIST data
        train_set = MyMNIST(file_path, train=True, download=True, transform=train_transform)
        unlabeled_set = MyMNIST(file_path, train=True, download=True, transform=test_transform)
        test_set = MyMNIST(file_path, train=False, download=True, transform=test_transform)
    elif args.dataset == 'SVHN':
        file_path = args.data_path + '/svhn/'
        train_set = MySVHN(file_path, split='train', download=True, transform=train_transform)
        unlabeled_set = MySVHN(file_path, split='train', download=True, transform=test_transform)
        test_set = MySVHN(file_path, split='test', download=True, transform=test_transform)
    elif args.dataset == 'ImageNet50':
        # Load Preprocessed IN-classes & indices; 50 classes were randomly selected
        index_path = args.data_path + '/ImageNet50/class_indice_dict.pickle'
        with open(index_path, 'rb') as f:
            class_indice_dict = pickle.load(f)
        print(class_indice_dict.keys()) #['in_class', 'in_indices', 'in_indices_test', 'ood_indices']

        file_path = '/data/pdm102207/imagenet/'
        train_set = MyImageNet(file_path+'train/', transform=train_transform)
        unlabeled_set = MyImageNet(file_path + 'train/', transform=test_transform)
        test_set = MyImageNet(file_path+ 'val/', transform=test_transform)
    elif args.dataset == 'TinyImageNet':
        # Load Preprocessed IN-classes & indices; 50 classes were randomly selected
        index_path = args.data_path + '/tiny-imagenet/class_indice_dict.pickle'
        with open(index_path, 'rb') as f:
            class_indice_dict = pickle.load(f)
        print(class_indice_dict.keys()) #['in_class', 'in_indices', 'in_indices_test', 'ood_indices']

        # Load the TinyImageNet dataset after creating/loading the indice dictionary
        file_path = args.data_path + '/tiny-imagenet/'
        train_set = MyTinyImageNet(file_path + 'train/', transform=train_transform)
        unlabeled_set = MyTinyImageNet(file_path + 'train/', transform=test_transform)
        test_set = MyTinyImageNet(file_path + 'val/', transform=test_transform)
    elif args.dataset in ['AGNEWS', 'IMDB', 'SST5']:
        # Load the text datasets
        if args.model == 'DistilBert':
            tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        elif args.model == 'Roberta':
            tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        
        # Initialize datasets for training, validation, and testing
        if args.dataset == 'SST5':
            file_path = args.data_path + '/sst5/'
            train_set = MySST5Dataset(file_path+'train.tsv', tokenizer=tokenizer, max_length=128)
            test_set = MySST5Dataset(file_path+'test.tsv', tokenizer=tokenizer, max_length=128)
            unlabeled_set = MySST5Dataset(file_path+'train.tsv', tokenizer=tokenizer, max_length=128)
        elif args.dataset == 'IMDB':
            train_set = MyIMDBDataset(split='train', tokenizer=tokenizer, max_length=128)
            test_set = MyIMDBDataset(split='test', tokenizer=tokenizer, max_length=128)
            unlabeled_set = MyIMDBDataset(split='train', tokenizer=tokenizer, max_length=128)
        else: # AGNEWS
            train_set = MyAGNewsDataset(split='train', tokenizer=tokenizer, max_length=128)
            test_set = MyAGNewsDataset(split='test', tokenizer=tokenizer, max_length=128)
            unlabeled_set = MyAGNewsDataset(split='train', tokenizer=tokenizer, max_length=128)

    # Class split configuration
    if args.dataset in ['CIFAR10', 'SVHN']:
        args.input_size = 32 * 32 * 3
        # Set total number of classes (including OOD)
        args.n_class = 10  # Total number of classes in the dataset
        args.target_list = list(range(10))  # All classes (for closed set)
        args.num_IN_class = 10  # Number of in-distribution classes
        
        if args.openset:
            # Settings for open set learning
            args.target_lists = [[4, 2, 5, 7], [7, 1, 2, 5], [6, 4, 3, 2], [8, 9, 1, 3], [2, 9, 5, 3], [3, 6, 4, 7]]
            args.target_list = args.target_lists[trial]
            args.num_IN_class = len(args.target_list)  # Number of in-distribution classes
        
        # Calculate untarget_list (classes that will be treated as OOD)
        args.untarget_list = list(np.setdiff1d(list(range(0, 10)), list(args.target_list)))

    elif args.dataset == 'AGNEWS':
        args.input_size = 128  # Sequence length
        # Set total number of classes (including OOD)
        args.n_class = 4  # Total number of classes in AGNEWS
        args.target_list = list(range(4))  # All classes (for closed set)
        args.num_IN_class = 4  # Number of in-distribution classes
        
        if args.openset:
            # Define different class splits for trials
            args.target_lists = [[0, 1], [2, 3], [0, 2], [1, 3]]
            if trial >= len(args.target_lists):
                print(f"Warning: Trial {trial} exceeds available target lists. Using trial % len(target_lists).")
            args.target_list = args.target_lists[trial % len(args.target_lists)]
            args.num_IN_class = len(args.target_list)  # Update number of in-distribution classes
        
        # Calculate untarget_list (classes that will be treated as OOD)
        args.untarget_list = list(np.setdiff1d(list(range(args.n_class)), list(args.target_list)))
    
    elif args.dataset == 'SST5':
        args.input_size = 128
        # Set total number of classes (including OOD)
        args.n_class = 5  # Total number of classes in SST5
        args.target_list = list(range(5))  # All classes (for closed set)
        args.num_IN_class = 5  # Number of in-distribution classes
        
        if args.openset:
            # Define different class combinations for trials
            args.target_lists = [
                [0, 2], [0, 3], [0, 4], [0, 5],
                [1, 2], [1, 3], [1, 4],
                [2, 3], [2, 4],
                [3, 4]
            ]
            if trial >= len(args.target_lists):
                print(f"Warning: Trial {trial} exceeds available target lists. Using trial % len(target_lists).")
            args.target_list = args.target_lists[trial % len(args.target_lists)]
            args.num_IN_class = len(args.target_list)  # Update number of in-distribution classes
        
        # Calculate untarget_list (classes that will be treated as OOD)
        args.untarget_list = list(np.setdiff1d(list(range(args.n_class)), list(args.target_list)))

    elif args.dataset == 'IMDB':
        args.input_size = 128
        # Set total number of classes (including OOD)
        args.n_class = 2  # Total number of classes in IMDB
        args.target_list = list(range(2))  # All classes (for closed set)
        args.num_IN_class = 2  # Number of in-distribution classes
        
        if args.openset:
            # Define different class splits for trials
            args.target_lists = [[0], [1], [0], [1]]  # One class as in-distribution at a time
            if trial >= len(args.target_lists):
                print(f"Warning: Trial {trial} exceeds available target lists. Using trial % len(target_lists).")
            args.target_list = args.target_lists[trial % len(args.target_lists)]
            args.num_IN_class = len(args.target_list)  # Update number of in-distribution classes
        
        # Calculate untarget_list (classes that will be treated as OOD)
        args.untarget_list = list(np.setdiff1d(list(range(args.n_class)), list(args.target_list)))

    elif args.dataset == 'MNIST':
        args.input_size = 28 * 28 * 1
        args.n_class = 10  # Total number of classes in MNIST
        args.target_list = list(range(10))  # All classes (for closed set)
        args.num_IN_class = 10  # Number of in-distribution classes
        
        if args.openset:
            # Settings for open set learning
            args.target_lists = [[4, 2, 5, 7], [7, 1, 2, 5], [6, 4, 3, 2], [8, 9, 1, 3], [2, 9, 5, 3], [3, 6, 4, 7]]
            args.target_list = args.target_lists[trial]
            args.num_IN_class = len(args.target_list)  # Number of in-distribution classes
        
        # Calculate untarget_list (classes that will be treated as OOD)
        args.untarget_list = list(np.setdiff1d(list(range(0, 10)), list(args.target_list)))
    
    elif args.dataset == 'CIFAR100':
        args.input_size = 32 * 32 * 3
        args.n_class = 100  # Total number of classes in CIFAR100
        args.target_list = list(range(100))  # All classes (for closed set)
        args.num_IN_class = 100  # Number of in-distribution classes
        
        if args.openset:
            # Settings for open set learning
            args.target_lists = [[69,  8, 86, 18, 68, 30, 75,  3, 63, 76, 72,  7, 50, 81, 46, 89, 22,
            93, 62, 21, 33, 98, 82, 20, 60,  5, 77,  1, 74, 88, 57, 34, 43, 27, 66, 83, 25, 48,  4, 55], \
                            [33, 10, 74, 72, 88, 47, 27, 68, 60, 75, 45, 79, 92, 35, 86, 50, 18,
            61, 49, 29, 23, 30, 67, 73, 82, 94, 13, 37, 39, 26, 62, 22, 90, 53, 89, 11,  3, 20, 70, 96], \
                            [70, 28, 60, 22, 39, 35, 73, 13, 74, 10,  2, 16, 80, 53, 67, 66, 78,
            46, 26, 71, 43, 38, 42, 14, 50, 77, 20, 48, 52,  8, 54, 58, 91,  5, 25, 90, 61, 11, 59, 55], \
                            [ 7, 93, 37, 84, 57, 99, 10, 75, 54, 42, 26, 27, 47, 52, 61, 86, 60,
            90,  1,  0, 98, 87, 94, 74, 56, 91, 23, 97, 30, 17, 53, 12, 76, 11, 25, 65, 96,  3, 45, 8], \
                            [ 0,  1,  4,  5,  7,  9, 12, 19, 21, 22, 23, 24, 38, 41, 42, 43, 46,
            47, 48, 51, 55, 59, 60, 62, 68, 73, 75, 78, 79, 80, 81, 85, 86, 90,91, 94, 95, 96, 97, 98]]
            args.target_list = args.target_lists[trial]
            args.num_IN_class = len(args.target_list)  # Number of in-distribution classes
        
        # Calculate untarget_list (classes that will be treated as OOD)
        args.untarget_list = list(np.setdiff1d(list(range(0, 100)), list(args.target_list)))
    
    elif args.dataset == 'ImageNet50' or args.dataset == 'TinyImageNet':    
        if args.openset:
            args.input_size = 64 * 64 * 3
            args.target_list = class_indice_dict['in_class']  # SEED 1
            args.in_indices = class_indice_dict['in_indices']
            args.in_test_indices = class_indice_dict['in_indices_test']
            # Create open set if args.openset is True
            args.ood_indices = random.sample(list(np.setdiff1d(list(range(0, len(train_set))), list(args.in_indices))),
                                       round(1.5 * len(args.in_indices)))
            args.untarget_list = list(np.setdiff1d(list(range(0, 1000)), list(args.target_list)))
            args.num_IN_class = 50  # For tinyimagenet
        else:
            args.input_size = 64 * 64 * 3
            args.target_list = list(range(0, 200))
            args.in_indices = np.concatenate([class_indice_dict['in_indices'], class_indice_dict['ood_indices']])
            args.in_test_indices = class_indice_dict['in_indices_test']
            # If args.openset is False, use all data (no open set creation)
            args.ood_indices = []  # No out-of-distribution samples
            args.untarget_list = []  # No unlabeled classes, all classes participate in training
            args.num_IN_class = 200  # For tinyimagenet

    # Class conversion for different dataset types
    if args.dataset in ['CIFAR10', 'CIFAR100', 'MNIST', 'SVHN']: 
        # For standard image datasets
        for i, c in enumerate(args.untarget_list):
            # Mark untarget classes with temporary value (args.n_class)
            train_set.targets[np.where(train_set.targets == c)[0]] = int(args.n_class)
            test_set.targets[np.where(test_set.targets == c)[0]] = int(args.n_class)

        # Sort target classes and relabel them from 0 to len(target_list)-1
        args.target_list.sort()
        for i, c in enumerate(args.target_list):
            train_set.targets[np.where(train_set.targets == c)[0]] = i
            test_set.targets[np.where(test_set.targets == c)[0]] = i

        # Relabel OOD classes to num_IN_class
        train_set.targets[np.where(train_set.targets == int(args.n_class))[0]] = int(args.num_IN_class)
        test_set.targets[np.where(test_set.targets == int(args.n_class))[0]] = int(args.num_IN_class)

    elif args.dataset in ['SST5']:
        # SST5特殊处理
        print("开始转换SST5数据集类别...")
        print(f"原始目标类别: {args.target_list}")
        print(f"原始非目标类别: {args.untarget_list}")
        
        # 确保targets属性存在
        if not hasattr(train_set, 'targets'):
            print("错误: 训练集没有'targets'属性。请检查数据集类的实现。")
            raise AttributeError("训练集中没有找到'targets'属性")
        if not hasattr(test_set, 'targets'):
            print("错误: 测试集没有'targets'属性。请检查数据集类的实现。")
            raise AttributeError("测试集中没有找到'targets'属性")
        
        # 转换为numpy数组
        train_targets = np.array(train_set.targets)
        test_targets = np.array(test_set.targets)
        
        # 打印转换前的原始类别分布
        train_unique, train_counts = np.unique(train_targets, return_counts=True)
        test_unique, test_counts = np.unique(test_targets, return_counts=True)
        print("转换前的原始类别分布:")
        print("训练集:", list(zip(train_unique, train_counts)))
        print("测试集:", list(zip(test_unique, test_counts)))
        
        # 创建新的目标数组，初始化为-1
        new_train_targets = np.full_like(train_targets, -1)
        new_test_targets = np.full_like(test_targets, -1)
        
        # 检查SST5数据集类别是否从0或1开始
        # 通常SST5的类别应该是0,1,2,3,4，但有时可能是1,2,3,4,5
        min_class = min(np.min(train_targets), np.min(test_targets))
        class_offset = min_class  # 如果最小类别是1，需要减1来调整
        
        if min_class > 0:
            print(f"注意: SST5类别从{min_class}开始而不是0，应用偏移量调整。")
        
        if args.openset:
            # 在开放集设置下:
            # 1. 为每个目标(in-distribution)类别分配新的标签
            for i, c in enumerate(args.target_list):
                # 调整类别以匹配数据集实际编码
                adjusted_class = c + class_offset
                print(f"将原始类别 {adjusted_class} 映射到新类别 {i}")
                new_train_targets[train_targets == adjusted_class] = i
                new_test_targets[test_targets == adjusted_class] = i
            
            # 2. 将所有非目标类别标记为OOD(args.num_IN_class)
            for c in args.untarget_list:
                # 调整类别以匹配数据集实际编码
                adjusted_class = c + class_offset
                print(f"将原始类别 {adjusted_class} 标记为OOD类别 {args.num_IN_class}")
                new_train_targets[train_targets == adjusted_class] = args.num_IN_class
                new_test_targets[test_targets == adjusted_class] = args.num_IN_class
        else:
            # 在闭集设置下:
            # 直接转换所有类别(不需要OOD)
            for i, c in enumerate(range(args.n_class)):
                # 调整类别以匹配数据集实际编码
                adjusted_class = c + class_offset
                new_train_targets[train_targets == adjusted_class] = i
                new_test_targets[test_targets == adjusted_class] = i
        
        # 确保所有样本都被分配了新标签
        if np.any(new_train_targets == -1) or np.any(new_test_targets == -1):
            unassigned_train = np.sum(new_train_targets == -1)
            unassigned_test = np.sum(new_test_targets == -1)
            print(f"错误: 有{unassigned_train}个训练样本和{unassigned_test}个测试样本未被分配类别!")
            # 打印未分配样本的原始类别
            if unassigned_train > 0:
                unassigned_classes = np.unique(train_targets[new_train_targets == -1])
                print(f"未分配的训练样本原始类别: {unassigned_classes}")
            if unassigned_test > 0:
                unassigned_classes = np.unique(test_targets[new_test_targets == -1])
                print(f"未分配的测试样本原始类别: {unassigned_classes}")
            
            # 将未分配的样本作为OOD处理
            if args.openset:
                print("将未分配的样本作为OOD处理")
                new_train_targets[new_train_targets == -1] = args.num_IN_class
                new_test_targets[new_test_targets == -1] = args.num_IN_class
            else:
                # 在闭集情况下，这不应该发生
                raise ValueError("在闭集设置中发现未分配的样本，请检查类别转换逻辑!")
        
        # 更新数据集的targets
        train_set.targets = new_train_targets
        test_set.targets = new_test_targets
        
        # 打印转换后的类别分布
        train_unique, train_counts = np.unique(train_set.targets, return_counts=True)
        test_unique, test_counts = np.unique(test_set.targets, return_counts=True)
        print("转换后的类别分布:")
        print("训练集:", list(zip(train_unique, train_counts)))
        print("测试集:", list(zip(test_unique, test_counts)))
        
        # 验证类别转换
        if args.openset:
            in_classes = np.unique(train_set.targets[train_set.targets < args.num_IN_class])
            expected_classes = np.arange(args.num_IN_class)
            
            if not np.array_equal(np.sort(in_classes), expected_classes):
                print(f"警告: 转换后的in-distribution类别{in_classes}与预期{expected_classes}不符!")
            
            in_count = np.sum(train_set.targets < args.num_IN_class)
            ood_count = np.sum(train_set.targets == args.num_IN_class)
            print(f"训练集 IN 样本数: {in_count}, OOD 样本数: {ood_count}")

    elif args.dataset in ['AGNEWS']:
        # 文本数据集特殊处理
        print("开始转换文本数据集类别...")
        print(f"原始目标类别: {args.target_list}")
        print(f"原始非目标类别: {args.untarget_list}")
        
        # 确保targets属性存在并转为numpy数组
        if not hasattr(train_set, 'targets'):
            print("错误: 训练集没有'targets'属性。请检查数据集类的实现。")
            raise AttributeError("训练集中没有找到'targets'属性")
        if not hasattr(test_set, 'targets'):
            print("错误: 测试集没有'targets'属性。请检查数据集类的实现。")
            raise AttributeError("测试集中没有找到'targets'属性")
        
        # 转换为numpy数组以便进行操作
        if not isinstance(train_set.targets, np.ndarray):
            train_set.targets = np.array(train_set.targets)
        if not isinstance(test_set.targets, np.ndarray):
            test_set.targets = np.array(test_set.targets)
        
        # 打印转换前的统计信息
        train_unique, train_counts = np.unique(train_set.targets, return_counts=True)
        test_unique, test_counts = np.unique(test_set.targets, return_counts=True)
        print("转换前的类别分布:")
        print("训练集:", list(zip(train_unique, train_counts)))
        print("测试集:", list(zip(test_unique, test_counts)))
        
        # 1. 首先，明确要标记为OOD的类别
        # 非目标类别(untarget_list)在开放集情况下应该被标记为OOD
        if args.openset:
            # 临时使用一个特殊值标记OOD类别
            temp_ood_marker = int(args.n_class) + 999  # 使用一个不可能是原始类别的值
            
            # 标记非目标类别为OOD(使用临时标记)
            for c in args.untarget_list:
                train_mask = (train_set.targets == c)
                test_mask = (test_set.targets == c)
                train_set.targets[train_mask] = temp_ood_marker
                test_set.targets[test_mask] = temp_ood_marker
            
            # 2. 重新映射目标类别(从0开始)
            args.target_list.sort()  # 确保目标列表有序
            for i, c in enumerate(args.target_list):
                train_mask = (train_set.targets == c)
                test_mask = (test_set.targets == c)
                train_set.targets[train_mask] = i
                test_set.targets[test_mask] = i
            
            # 3. 将临时OOD标记替换为最终OOD类别索引(num_IN_class)
            train_set.targets[train_set.targets == temp_ood_marker] = int(args.num_IN_class)
            test_set.targets[test_set.targets == temp_ood_marker] = int(args.num_IN_class)
        
        # 打印转换后的统计信息
        train_unique, train_counts = np.unique(train_set.targets, return_counts=True)
        test_unique, test_counts = np.unique(test_set.targets, return_counts=True)
        print("转换后的类别分布:")
        print("训练集:", list(zip(train_unique, train_counts)))
        print("测试集:", list(zip(test_unique, test_counts)))
    
    elif args.dataset in ['IMDB']:
        # For text datasets, ensure proper handling of targets
        if hasattr(train_set, 'targets') and hasattr(test_set, 'targets'):
            # Convert targets to numpy arrays for easier manipulation if they're not already
            if not isinstance(train_set.targets, np.ndarray):
                train_set.targets = np.array(train_set.targets)
            if not isinstance(test_set.targets, np.ndarray):
                test_set.targets = np.array(test_set.targets)
                
            # Mark untarget classes as OOD with temporary value
            for c in args.untarget_list:
                train_set.targets[train_set.targets == c] = int(args.n_class)
                test_set.targets[test_set.targets == c] = int(args.n_class)
            
            # Sort target list and relabel target classes from 0 to len(target_list)-1
            args.target_list.sort()
            for i, c in enumerate(args.target_list):
                train_set.targets[train_set.targets == c] = i
                test_set.targets[test_set.targets == c] = i
            
            # Relabel OOD classes to num_IN_class
            # This creates a dedicated OOD class with index equal to num_IN_class
            train_set.targets[train_set.targets == int(args.n_class)] = int(args.num_IN_class)
            test_set.targets[test_set.targets == int(args.n_class)] = int(args.num_IN_class)
        else:
            # Handling for text datasets with different structure
            print("Warning: Text dataset structure does not have 'targets' attribute.")
            print("Please implement custom target conversion for this dataset structure.")
            # Implement custom conversion based on actual dataset structure
            # This might involve modifying the dataset class to support this conversion

    elif args.dataset == 'ImageNet50' or args.dataset == 'TinyImageNet':
        if args.openset:
            args.target_list.sort()
            class_covert_dict = {}
            for i, c in enumerate(args.target_list):
                class_covert_dict[c] = i  # {original_class : new_class_index}

            # Convert in-distribution class labels
            for idx in args.in_indices:
                train_set.targets[idx] = class_covert_dict[train_set.targets[idx]]

            # Mark OOD samples
            for idx in args.ood_indices:
                train_set.targets[idx] = int(args.num_IN_class)

            # Process test samples
            in_test_indices_set = set(args.in_test_indices)
            total_test_samples = len(test_set.targets)
            
            for idx in range(total_test_samples):
                original_label = test_set.targets[idx]
                if idx in in_test_indices_set:
                    # If index is in in_test_indices, apply class mapping
                    if original_label in class_covert_dict:
                        test_set.targets[idx] = class_covert_dict[original_label]
                    else:
                        # If original label is not in class_covert_dict, mark as OOD
                        test_set.targets[idx] = int(args.num_IN_class)
                else:
                    # If index is not in in_test_indices, mark as OOD
                    test_set.targets[idx] = int(args.num_IN_class)

    # Copy train_set targets to unlabeled_set (with proper handling of array types)
    unlabeled_set.targets = train_set.targets.copy() if isinstance(train_set.targets, np.ndarray) else train_set.targets.copy()

    # Split Check and Reporting
    print("Target classes: ", args.target_list)
    
    if args.dataset in ['CIFAR10', 'CIFAR100', 'MNIST', 'SVHN']:
        if args.method == 'EPIG':
            uni, cnt = np.unique(np.array(unlabeled_set.targets), return_counts=True)
            print("Train, # samples per class")
            cnt -= args.target_per_class
            print(uni, cnt)
        else:    
            uni, cnt = np.unique(np.array(unlabeled_set.targets), return_counts=True)
            print("Train, # samples per class")
            print(uni, cnt)
        uni, cnt = np.unique(np.array(test_set.targets), return_counts=True)
        print("Test, # samples per class")
        print(uni, cnt)
    
    elif args.dataset == 'ImageNet50' or args.dataset == 'TinyImageNet':
        uni, cnt = np.unique(np.array(unlabeled_set.targets[args.in_indices]), return_counts=True)
        print("Train IN, # samples per class")
        print(uni, cnt)
        uni, cnt = np.unique(np.array(unlabeled_set.targets[args.ood_indices]), return_counts=True)
        print("Train OOD (Sampled), # samples per class")
        print(uni, cnt)
        uni, cnt = np.unique(np.array(test_set.targets[args.in_test_indices]), return_counts=True)
        print("Test, # samples per class")
        print(uni, cnt)

    elif args.dataset in ['AGNEWS', 'IMDB', 'SST5']:
        # 文本数据集报告
        uni, cnt = np.unique(np.array(unlabeled_set.targets), return_counts=True)
        print("训练集, 每个类别的样本数")
        print(list(zip(uni, cnt)))
        
        # 分别报告in-distribution和OOD类别
        if args.openset:
            in_mask = np.array([t < args.num_IN_class for t in unlabeled_set.targets])
            ood_mask = np.array([t == args.num_IN_class for t in unlabeled_set.targets])
            
            in_count = np.sum(in_mask)
            ood_count = np.sum(ood_mask)
            
            in_classes = np.unique(unlabeled_set.targets[in_mask])
            print(f"训练集 IN 样本: {in_count}, 类别: {in_classes}")
            print(f"训练集 OOD 样本: {ood_count}")
            
            # 验证是否所有目标类别都有样本
            if len(in_classes) != len(args.target_list):
                print(f"警告: 目标类别数({len(args.target_list)})与实际in-distribution类别数({len(in_classes)})不匹配!")
        
        uni, cnt = np.unique(np.array(test_set.targets), return_counts=True)
        print("测试集, 每个类别的样本数")
        print(list(zip(uni, cnt)))
        
        # 分别报告测试集的in-distribution和OOD类别
        if args.openset:
            in_mask = np.array([t < args.num_IN_class for t in test_set.targets])
            ood_mask = np.array([t == args.num_IN_class for t in test_set.targets])
            
            in_count = np.sum(in_mask)
            ood_count = np.sum(ood_mask)
            
            in_classes = np.unique(test_set.targets[in_mask])
            print(f"测试集 IN 样本: {in_count}, 类别: {in_classes}")
            print(f"测试集 OOD 样本: {ood_count}")
    # elif args.dataset in ['AGNEWS', 'IMDB', 'SST5']:
    #     # Reporting for text datasets
    #     uni, cnt = np.unique(np.array(unlabeled_set.targets), return_counts=True)
    #     print("Train, # samples per class")
    #     print(uni, cnt)
        
    #     # Separate reporting for in-distribution and OOD classes
    #     in_mask = np.array([t < args.num_IN_class for t in unlabeled_set.targets])
    #     ood_mask = np.array([t == args.num_IN_class for t in unlabeled_set.targets])
        
    #     in_count = np.sum(in_mask)
    #     ood_count = np.sum(ood_mask)
    #     print(f"Train IN samples: {in_count}, OOD samples: {ood_count}")
        
    #     uni, cnt = np.unique(np.array(test_set.targets), return_counts=True)
    #     print("Test, # samples per class")
    #     print(uni, cnt)
        
    #     # Separate reporting for test in-distribution and OOD classes
    #     in_mask = np.array([t < args.num_IN_class for t in test_set.targets])
    #     ood_mask = np.array([t == args.num_IN_class for t in test_set.targets])
        
    #     in_count = np.sum(in_mask)
    #     ood_count = np.sum(ood_mask)
    #     print(f"Test IN samples: {in_count}, OOD samples: {ood_count}")
    
    return train_set, unlabeled_set, test_set

def get_superclass_list(dataset):
    if dataset == 'CIFAR10':
        return CIFAR10_SUPERCLASS
    elif dataset == 'CIFAR100':
        return CIFAR100_SUPERCLASS
    elif dataset == 'ImageNet50':
        return IMAGENET_SUPERCLASS
    else:
        raise NotImplementedError()

def get_subclass_dataset(dataset, classes):
    if not isinstance(classes, list):
        classes = [classes]

    indices = []
    for idx, tgt in enumerate(dataset.targets):
        if tgt in classes:
            indices.append(idx)

    dataset = Subset(dataset, indices)
    return dataset

def get_sub_train_dataset(args, dataset, L_index, O_index, U_index, Q_index, initial=False):
    classes = args.target_list
    budget = args.n_initial
    ood_rate = args.ood_rate

    if initial:
        if args.dataset in ['CIFAR10', 'CIFAR100', 'MNIST', 'SVHN', 'AGNEWS', 'IMDB', 'SST5']:
            if args.openset:
                # Handle text datasets differently
                if args.dataset in ['AGNEWS', 'IMDB', 'SST5']:
                    L_total = [dataset[i]['index'] for i in range(len(dataset)) if dataset[i]['labels'] < len(classes)]
                    O_total = [dataset[i]['index'] for i in range(len(dataset)) if dataset[i]['labels'] >= len(classes)]
                else:
                    # Handle image datasets
                    L_total = [dataset[i][2] for i in range(len(dataset)) if dataset[i][1] < len(classes)]
                    O_total = [dataset[i][2] for i in range(len(dataset)) if dataset[i][1] >= len(classes)]

                # Calculate number of OOD samples based on ood_rate
                n_ood = round(ood_rate * (len(L_total) + len(O_total)))

                # Check if we have enough OOD samples
                if n_ood > len(O_total):
                    print('The currently designed number of OOD samples is ' + str(n_ood) + ', but the actual number of OOD samples in the dataset is only ' + str(len(O_total)) + '.')
                    print('Using all OOD data and adjusting the ID data to maintain the OOD rate.')
                    n_ood = len(O_total)
                    n_id = round(len(O_total)/ood_rate - len(O_total))
                    # Make sure we don't sample more than available
                    n_id = min(n_id, len(L_total))
                    L_total = random.sample(L_total, n_id)
                else:
                    # Sample OOD based on calculated number
                    O_total = random.sample(O_total, n_ood)
                
                print("# Total in: {}, ood: {}".format(len(L_total), len(O_total)))
                
                # Initialize indices for labeled and unlabeled sets
                if len(L_total) < budget - int(budget * ood_rate):
                    print("Warning: Not enough in-distribution samples for requested budget.")
                    L_index = L_total  # Use all available in-distribution samples
                else:
                    L_index = random.sample(L_total, budget - int(budget * ood_rate))
                
                if len(O_total) < int(budget * ood_rate):
                    print("Warning: Not enough OOD samples for requested budget.")
                    O_index = O_total  # Use all available OOD samples
                else:
                    O_index = random.sample(O_total, int(budget * ood_rate))
                
                # Create unlabeled set
                U_index = list(set(L_total + O_total) - set(L_index) - set(O_index))
                
                # Report statistics
                if args.method == 'EPIG':
                    print("# Labeled in: {}, ood: {}, Unlabeled: {}".format(
                        len(L_index), 
                        len(O_index), 
                        len(U_index) - args.num_IN_class * args.target_per_class
                    ))
                else:
                    print("# Labeled in: {}, ood: {}, Unlabeled: {}".format(
                        len(L_index), 
                        len(O_index), 
                        len(U_index)
                    ))
            else:
                # No open set (closed set scenario)
                ood_rate = 0
                O_index = []  # Initialize as empty list
                
                # Handle text datasets differently
                if args.dataset in ['AGNEWS', 'IMDB', 'SST5']:
                    L_total = [dataset[i]['index'] for i in range(len(dataset)) if dataset[i]['labels'] < len(classes)]
                else:
                    L_total = [dataset[i][2] for i in range(len(dataset)) if dataset[i][1] < len(classes)]
                
                O_total = []
                n_ood = 0
                print("# Total in: {}, ood: {}".format(len(L_total), len(O_total)))

                # Make sure we don't sample more than available
                budget_adjusted = min(int(budget), len(L_total))
                if budget_adjusted < budget:
                    print(f"Warning: Requested budget {budget} exceeds available samples {len(L_total)}.")
                    print(f"Using all {len(L_total)} available samples.")
                
                L_index = random.sample(L_total, budget_adjusted)
                U_index = list(set(L_total) - set(L_index))
                
                if args.method == 'EPIG':
                    print("# Labeled in: {}, ood: {}, Unlabeled: {}".format(
                        len(L_index), 
                        len(O_index), 
                        len(U_index) - args.num_IN_class * args.target_per_class
                    ))
                else:
                    print("# Labeled in: {}, ood: {}, Unlabeled: {}".format(
                        len(L_index), 
                        len(O_index), 
                        len(U_index)
                    ))

        elif args.dataset == 'ImageNet50' or args.dataset == 'TinyImageNet':
            # ImageNet/TinyImageNet processing
            L_total = [dataset[i][2] for i in args.in_indices]
            O_total = [dataset[i][2] for i in args.ood_indices]

            if args.openset:
                n_ood = round(len(L_total) * (ood_rate / (1 - ood_rate)))
                # Check if we have enough OOD samples
                if n_ood > len(O_total):
                    print('Warning: Not enough OOD samples. Using all available OOD samples.')
                    n_ood = len(O_total)
            else:
                n_ood = 0
                ood_rate = 0
            
            # Sample OOD samples if needed
            if n_ood > 0:
                O_total = random.sample(O_total, n_ood)
            
            print("# Total in: {}, ood: {}".format(len(L_total), len(O_total)))

            # Calculate budget
            in_budget = int(budget * (1 - ood_rate))
            ood_budget = int(budget * ood_rate)
            
            # Check if we have enough samples for the budget
            if in_budget > len(L_total):
                print(f"Warning: Not enough in-distribution samples. Using all {len(L_total)} available samples.")
                in_budget = len(L_total)
            
            if ood_budget > len(O_total):
                print(f"Warning: Not enough OOD samples. Using all {len(O_total)} available samples.")
                ood_budget = len(O_total)
            
            L_index = random.sample(L_total, in_budget)
            O_index = random.sample(O_total, ood_budget)
            U_index = list(set(L_total + O_total) - set(L_index) - set(O_index))
            
            print("# Labeled in: {}, ood: {}, Unlabeled: {}".format(len(L_index), len(O_index), len(U_index)))
        
        return L_index, O_index, U_index
    
    else:
        # Non-initial round (update after query)
        Q_index = list(Q_index)  # Ensure Q_index is a list
        
        # Get labels for query indices
        if args.dataset in ['AGNEWS', 'IMDB', 'SST5']:
            Q_label = [dataset[i]['labels'] for i in Q_index]
        else:
            Q_label = [dataset[i][1] for i in Q_index]

        # Separate in-distribution and OOD queries
        in_Q_index, ood_Q_index = [], []
        for i, c in enumerate(Q_label):
            if c < len(classes):
                in_Q_index.append(Q_index[i])
            else:
                ood_Q_index.append(Q_index[i])
        
        print("# query in: {}, ood: {}".format(len(in_Q_index), len(ood_Q_index)))
        
        # Update indices
        L_index = list(L_index) + in_Q_index  # Ensure L_index is a list before addition
        print("# Now labelled in: {}".format(len(L_index)))
        
        O_index = list(O_index) + ood_Q_index  # Ensure O_index is a list before addition
        U_index = list(set(U_index) - set(Q_index))
        
        return L_index, O_index, U_index, len(in_Q_index)

def get_sub_test_dataset(args, dataset):
    classes = args.target_list
    if args.dataset in ['CIFAR10', 'CIFAR100', 'MNIST', 'SVHN']: # add 'MNIST'
        labeled_index = [dataset[i][2] for i in range(len(dataset)) if dataset[i][1] < len(classes)]
    elif args.dataset == 'ImageNet50' or args.dataset == 'TinyImageNet':
        labeled_index = [dataset[i][2] for i in args.in_test_indices]
    if args.dataset in ['AGNEWS', 'IMDB', 'SST5']:
        labeled_index = [dataset[i]['index'] for i in range(len(dataset)) if dataset[i]['labels'] < len(classes)]    
    return labeled_index