import numpy as np
import torch
import random
from torch.utils.data.dataset import Subset
from ownDatasets.tinyimagenet import MyTinyImageNet
from ownDatasets.cifar10 import MyCIFAR10
from ownDatasets.cifar100 import MyCIFAR100
from ownDatasets.mnist import MyMNIST
from ownDatasets.svhn import MySVHN
from ownDatasets.agnews import MyAGNewsDataset
from ownDatasets.imdb import MyIMDBDataset
from ownDatasets.sst5 import MySST5Dataset
import torchvision.transforms as T
from transformers import DistilBertTokenizer
from transformers import RobertaTokenizer
from datasets import load_dataset
from torchvision.transforms import Lambda

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
    elif args.dataset == 'TinyImageNet':
        T_normalize = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    elif args.dataset == 'MNIST':
        T_normalize = T.Normalize([0.1307], [0.3081])  # Mean and std for MNIST
    elif args.dataset == 'SVHN':
        T_normalize = T.Normalize([0.4377, 0.4438, 0.4728], [0.1980, 0.2010, 0.1970])  # Mean and std for SVHN

    # Transform
    if args.dataset in ['CIFAR10', 'CIFAR100', 'SVHN']:
        train_transform = T.Compose([T.RandomHorizontalFlip(), T.RandomCrop(size=32, padding=4), T.ToTensor(), T_normalize])  #
        test_transform = T.Compose([T.ToTensor(), T_normalize])
    elif args.dataset == 'TinyImageNet':
        # train_transform = T.Compose([T.Resize(64), T.RandomCrop(64, padding=4), T.RandomHorizontalFlip(), T.ToTensor(), T_normalize])
        # test_transform = T.Compose([T.Resize(64), T.ToTensor(), T_normalize])
        train_transform = T.Compose([
            T.Resize(64),
            T.RandomCrop(64, padding=4),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            # Convert grayscale to RGB if needed
            Lambda(lambda x: x.repeat(3, 1, 1) if x.size(0) == 1 else x),
            T_normalize
        ])

        test_transform = T.Compose([
            T.Resize(64),
            T.ToTensor(),
            # Convert grayscale to RGB if needed
            Lambda(lambda x: x.repeat(3, 1, 1) if x.size(0) == 1 else x),
            T_normalize
        ])
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
        cifar10_dataset = load_dataset('cifar10')
        train_set = MyCIFAR10(cifar10_dataset['train'], transform=train_transform, imbalance_factor = args.imb_factor)
        unlabeled_set = MyCIFAR10(cifar10_dataset['train'], transform=test_transform, imbalance_factor = args.imb_factor)
        test_set = MyCIFAR10(cifar10_dataset['test'], transform=test_transform, imbalance_factor = args.imb_factor)
    elif args.dataset == 'CIFAR100':
        cifar100_dataset = load_dataset('cifar100')
        train_set = MyCIFAR100(cifar100_dataset['train'], transform=train_transform, imbalance_factor = args.imb_factor)
        unlabeled_set = MyCIFAR100(cifar100_dataset['train'], transform=test_transform, imbalance_factor = args.imb_factor)
        test_set = MyCIFAR100(cifar100_dataset['test'], transform=test_transform, imbalance_factor = args.imb_factor)
    elif args.dataset == 'MNIST':
        mnist_dataset = load_dataset('mnist')
        train_set = MyMNIST(mnist_dataset['train'], transform=train_transform)
        unlabeled_set = MyMNIST(mnist_dataset['train'], transform=test_transform)
        test_set = MyMNIST(mnist_dataset['test'], transform=test_transform)
    elif args.dataset == 'SVHN':
        svhn_dataset = load_dataset('svhn', name='cropped_digits')
        train_set = MySVHN(svhn_dataset['train'], transform=train_transform)
        unlabeled_set = MySVHN(svhn_dataset['train'], transform=test_transform)
        test_set = MySVHN(svhn_dataset['test'], transform=test_transform)
    elif args.dataset == 'TinyImageNet':
        # TinyImageNet is not directly available in Hugging Face datasets
        tiny_imagenet_dataset = load_dataset('zh-plus/tiny-imagenet')
        train_set = MyTinyImageNet(tiny_imagenet_dataset['train'], transform=train_transform, imbalance_factor = args.imb_factor)
        unlabeled_set = MyTinyImageNet(tiny_imagenet_dataset['train'], transform=test_transform, imbalance_factor = args.imb_factor)
        test_set = MyTinyImageNet(tiny_imagenet_dataset['valid'], transform=test_transform, imbalance_factor = args.imb_factor)
    elif args.dataset in ['AGNEWS', 'IMDB', 'SST5']:
        # Load the text datasets
        if args.model == 'DistilBert':
            tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        elif args.model == 'Roberta':
            tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        
        # Initialize datasets for training, validation, and testing
        if args.dataset == 'SST5':
            dataset = load_dataset('SetFit/sst5')
            train_set = MySST5Dataset(dataset['train'], tokenizer, imbalance_factor = args.imb_factor)
            test_set = MySST5Dataset(dataset['test'], tokenizer, imbalance_factor = args.imb_factor)
            unlabeled_set = MySST5Dataset(dataset['train'], tokenizer, imbalance_factor = args.imb_factor)
        elif args.dataset == 'IMDB':
            imdb_dataset = load_dataset('imdb')
            train_set = MyIMDBDataset(imdb_dataset['train'], tokenizer=tokenizer, max_length=128, imbalance_factor = args.imb_factor)
            test_set = MyIMDBDataset(imdb_dataset['test'], tokenizer=tokenizer, max_length=128, imbalance_factor = args.imb_factor)
            unlabeled_set = MyIMDBDataset(imdb_dataset['train'], tokenizer=tokenizer, max_length=128, imbalance_factor = args.imb_factor)
        else:  # AGNEWS
            agnews_dataset = load_dataset('ag_news')
            train_set = MyAGNewsDataset(agnews_dataset['train'], tokenizer=tokenizer, max_length=128, imbalance_factor = args.imb_factor)
            test_set = MyAGNewsDataset(agnews_dataset['test'], tokenizer=tokenizer, max_length=128, imbalance_factor = args.imb_factor)
            unlabeled_set = MyAGNewsDataset(agnews_dataset['train'], tokenizer=tokenizer, max_length=128, imbalance_factor = args.imb_factor)

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
    
    elif args.dataset == 'TinyImageNet':    
        if args.openset:
            args.input_size = 64 * 64 * 3
            args.target_list = random.sample(list(range(200)), int(200 * (1 - args.ood_rate))) # SEED 1 200*(1-args.ood_rate)
            args.untarget_list = list(np.setdiff1d(list(range(0, 200)), list(args.target_list)))
            args.num_IN_class = len(args.target_list)  # For tinyimagenet
        else:
            args.n_class = 200  # Total number of classes in TinyImageNet
            args.input_size = 64 * 64 * 3
            args.target_list = list(range(0, 200))
            args.untarget_list = []  # No unlabeled classes, all classes participate in training
            args.num_IN_class = 200  # For tinyimagenet

    # Class conversion for different dataset types
    if args.dataset in ['CIFAR10', 'CIFAR100', 'MNIST', 'SVHN', 'TinyImageNet']: 
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
    
    elif args.dataset in ['SST5', 'AGNEWS','IMDB']:
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

    # Copy train_set targets to unlabeled_set (with proper handling of array types)
    unlabeled_set.targets = train_set.targets.copy() if isinstance(train_set.targets, np.ndarray) else train_set.targets.copy()

    # Split Check and Reporting
    print("Target classes: ", args.target_list)
    
    if args.dataset in ['CIFAR10', 'CIFAR100', 'MNIST', 'SVHN', 'TinyImageNet', 'AGNEWS', 'IMDB', 'SST5']:
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
        if args.dataset in ['CIFAR10', 'CIFAR100', 'MNIST', 'SVHN', 'AGNEWS', 'IMDB', 'SST5', 'TinyImageNet']:
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
    if args.dataset in ['CIFAR10', 'CIFAR100', 'MNIST', 'SVHN', 'TinyImageNet']: # add 'MNIST'
        labeled_index = [dataset[i][2] for i in range(len(dataset)) if dataset[i][1] < len(classes)]
    if args.dataset in ['AGNEWS', 'IMDB', 'SST5']:
        labeled_index = [dataset[i]['index'] for i in range(len(dataset)) if dataset[i]['labels'] < len(classes)]    
    return labeled_index