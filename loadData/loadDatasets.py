import numpy as np
import random
from datasets import load_dataset
from transformers import DistilBertTokenizer, RobertaTokenizer

# Import custom dataset classes
from ownDatasets.tinyimagenet import MyTinyImageNet
from ownDatasets.cifar10 import MyCIFAR10
from ownDatasets.cifar100 import MyCIFAR100
from ownDatasets.mnist import MyMNIST
from ownDatasets.svhn import MySVHN
from ownDatasets.agnews import MyAGNewsDataset
from ownDatasets.imdb import MyIMDBDataset
from ownDatasets.sst5 import MySST5Dataset
from ownDatasets.dbpedia import MyDbpediaDataset
from ownDatasets.yelp import MyYelpDataset

# Import transforms
from .dataTransform import get_dataset_transforms

def get_dataset(args, trial):
    """
    Load and prepare datasets for training and testing.
    
    Args:
        args: Arguments containing dataset configuration
        trial: Trial number for open set learning
        
    Returns:
        A tuple of (train_set, unlabeled_set, test_set)
    """
    # Get dataset-specific transforms
    if args.dataset in ['CIFAR10', 'CIFAR100', 'MNIST', 'SVHN', 'TINYIMAGENET']:
        train_transform, test_transform = get_dataset_transforms(args.dataset)
    
# Dataset loading
    if args.dataset == 'CIFAR10':
        cifar10_dataset = load_dataset('cifar10')
        train_set = MyCIFAR10(cifar10_dataset['train'], transform=train_transform, imbalance_factor=args.imb_factor, method=args.method)
        unlabeled_set = MyCIFAR10(cifar10_dataset['train'], transform=test_transform, imbalance_factor=args.imb_factor, method=args.method)
        test_set = MyCIFAR10(cifar10_dataset['test'], transform=test_transform, imbalance_factor=args.imb_factor, method=args.method)
    elif args.dataset == 'CIFAR100':
        cifar100_dataset = load_dataset('cifar100')
        train_set = MyCIFAR100(cifar100_dataset['train'], transform=train_transform, imbalance_factor=args.imb_factor, method=args.method)
        unlabeled_set = MyCIFAR100(cifar100_dataset['train'], transform=test_transform, imbalance_factor=args.imb_factor, method=args.method)
        test_set = MyCIFAR100(cifar100_dataset['test'], transform=test_transform, imbalance_factor=args.imb_factor, method=args.method)
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
    elif args.dataset == 'TINYIMAGENET':
        # TinyImageNet is not directly available in Hugging Face datasets
        tiny_imagenet_dataset = load_dataset('zh-plus/tiny-imagenet')
        train_set = MyTinyImageNet(tiny_imagenet_dataset['train'], transform=train_transform, imbalance_factor=args.imb_factor, method=args.method)
        unlabeled_set = MyTinyImageNet(tiny_imagenet_dataset['train'], transform=test_transform, imbalance_factor=args.imb_factor, method=args.method)
        test_set = MyTinyImageNet(tiny_imagenet_dataset['valid'], transform=test_transform, imbalance_factor=args.imb_factor, method=args.method)
    elif args.textset:
        # Load the text datasets
        if args.model == 'DistilBert':
            tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        elif args.model == 'Roberta':
            tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        
        if args.dataset == 'SST5':
            sst5_dataset = load_dataset('SetFit/sst5')
            train_set = MySST5Dataset(sst5_dataset['train'], tokenizer, imbalance_factor=args.imb_factor)
            test_set = MySST5Dataset(sst5_dataset['test'], tokenizer, imbalance_factor=args.imb_factor)
            unlabeled_set = MySST5Dataset(sst5_dataset['train'], tokenizer, imbalance_factor=args.imb_factor)
        elif args.dataset == 'YELP':
            yelp_dataset = load_dataset("Yelp/yelp_review_full")
            train_set = MyYelpDataset(yelp_dataset['train'], tokenizer, imbalance_factor=args.imb_factor)
            test_set = MyYelpDataset(yelp_dataset['test'], tokenizer, imbalance_factor=args.imb_factor)
            unlabeled_set = MyYelpDataset(yelp_dataset['train'], tokenizer, imbalance_factor=args.imb_factor)
        elif args.dataset == 'IMDB':
            imdb_dataset = load_dataset('imdb')
            train_set = MyIMDBDataset(imdb_dataset['train'], tokenizer=tokenizer, imbalance_factor=args.imb_factor)
            test_set = MyIMDBDataset(imdb_dataset['test'], tokenizer=tokenizer, imbalance_factor=args.imb_factor)
            unlabeled_set = MyIMDBDataset(imdb_dataset['train'], tokenizer=tokenizer, imbalance_factor=args.imb_factor)
        elif args.dataset == 'DBPEDIA':
            dbpedia_dataset = load_dataset("fancyzhx/dbpedia_14")
            train_set = MyDbpediaDataset(dbpedia_dataset['train'],tokenizer=tokenizer, imbalance_factor=args.imb_factor)
            test_set = MyDbpediaDataset(dbpedia_dataset['test'],tokenizer=tokenizer, imbalance_factor=args.imb_factor)
            unlabeled_set = MyDbpediaDataset(dbpedia_dataset['train'],tokenizer=tokenizer, imbalance_factor=args.imb_factor)
        elif args.dataset == 'AGNEWS':
            agnews_dataset = load_dataset('ag_news')
            train_set = MyAGNewsDataset(agnews_dataset['train'], tokenizer=tokenizer, imbalance_factor=args.imb_factor)
            test_set = MyAGNewsDataset(agnews_dataset['test'], tokenizer=tokenizer, imbalance_factor=args.imb_factor)
            unlabeled_set = MyAGNewsDataset(agnews_dataset['train'], tokenizer=tokenizer, imbalance_factor=args.imb_factor)
        else:
            raise ValueError(f"Text dataset '{args.dataset}' is not supported. Please choose from the available text datasets.")
    else:
        raise ValueError(f"Dataset '{args.dataset}' is not supported. Please choose from the available datasets.")

    # Configure dataset settings based on type
    _configure_dataset_settings(args, trial)
    
    # Apply class conversion for different dataset types
    _apply_class_conversion(args, train_set, test_set, unlabeled_set)
    
    # Report split statistics
    _report_split_statistics(args, unlabeled_set, test_set)
    
    return train_set, unlabeled_set, test_set

def _configure_dataset_settings(args, trial):
    """
    Configure dataset settings including class splits for open set learning.
    
    Args:
        args: Arguments containing dataset configuration
        trial: Trial number for open set learning
    """
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
                [0, 2], [1, 3], [2, 4], [3, 4],
                [1, 4], [0, 3], [1, 2],
                [2, 3], [0, 4],
                [3, 4]
            ]
            if trial >= len(args.target_lists):
                print(f"Warning: Trial {trial} exceeds available target lists. Using trial % len(target_lists).")
            args.target_list = args.target_lists[trial % len(args.target_lists)]
            args.num_IN_class = len(args.target_list)  # Update number of in-distribution classes
        
        # Calculate untarget_list (classes that will be treated as OOD)
        args.untarget_list = list(np.setdiff1d(list(range(args.n_class)), list(args.target_list)))

    elif args.dataset == 'YELP':
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

    elif args.dataset == 'DBPEDIA':
        args.input_size = 128
        # Set total number of classes (including OOD)
        args.n_class = 14  # Total number of classes in SST5
        args.target_list = list(range(14))  # All classes (for closed set)
        args.num_IN_class = 14  # Number of in-distribution classes
        
        if args.openset:
            # Define different class combinations for trials
            args.target_lists = [[4, 2, 5, 7], [7, 1, 2, 5], [6, 4, 3, 2], [8, 9, 1, 3], [2, 9, 5, 3], [3, 6, 4, 7]]
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
            # Settings for open set learning (compact representation of the original lists)
            args.target_lists = [
                [69, 8, 86, 18, 68, 30, 75, 3, 63, 76, 72, 7, 50, 81, 46, 89, 22, 93, 62, 21, 33, 98, 82, 20, 60, 5, 77, 1, 74, 88, 57, 34, 43, 27, 66, 83, 25, 48, 4, 55], 
                [33, 10, 74, 72, 88, 47, 27, 68, 60, 75, 45, 79, 92, 35, 86, 50, 18, 61, 49, 29, 23, 30, 67, 73, 82, 94, 13, 37, 39, 26, 62, 22, 90, 53, 89, 11, 3, 20, 70, 96], 
                [70, 28, 60, 22, 39, 35, 73, 13, 74, 10, 2, 16, 80, 53, 67, 66, 78, 46, 26, 71, 43, 38, 42, 14, 50, 77, 20, 48, 52, 8, 54, 58, 91, 5, 25, 90, 61, 11, 59, 55], 
                [7, 93, 37, 84, 57, 99, 10, 75, 54, 42, 26, 27, 47, 52, 61, 86, 60, 90, 1, 0, 98, 87, 94, 74, 56, 91, 23, 97, 30, 17, 53, 12, 76, 11, 25, 65, 96, 3, 45, 8], 
                [0, 1, 4, 5, 7, 9, 12, 19, 21, 22, 23, 24, 38, 41, 42, 43, 46, 47, 48, 51, 55, 59, 60, 62, 68, 73, 75, 78, 79, 80, 81, 85, 86, 90, 91, 94, 95, 96, 97, 98]
            ]
            args.target_list = args.target_lists[trial]
            args.num_IN_class = len(args.target_list)  # Number of in-distribution classes
        
        # Calculate untarget_list (classes that will be treated as OOD)
        args.untarget_list = list(np.setdiff1d(list(range(0, 100)), list(args.target_list)))
    
    elif args.dataset == 'TINYIMAGENET':    
        if args.openset:
            random.seed(args.seed + trial) # set random seed
            args.input_size = 64 * 64 * 3
            args.target_list = random.sample(list(range(200)), int(200 * (1 - args.ood_rate))) # 200*(1-args.ood_rate)
            args.untarget_list = list(np.setdiff1d(list(range(0, 200)), list(args.target_list)))
            args.num_IN_class = len(args.target_list)  # For tinyimagenet
        else:
            args.n_class = 200  # Total number of classes in TinyImageNet
            args.input_size = 64 * 64 * 3
            args.target_list = list(range(0, 200))
            args.untarget_list = []  # No unlabeled classes, all classes participate in training
            args.num_IN_class = 200  # For tinyimagenet

def _apply_class_conversion(args, train_set, test_set, unlabeled_set):
    """
    Apply class conversion for different dataset types.
    
    Args:
        args: Arguments containing dataset configuration
        train_set: Training dataset
        test_set: Testing dataset
        unlabeled_set: Unlabeled dataset
    """

    if args.textset:
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

    else: # for image datasets
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

    # Copy train_set targets to unlabeled_set (with proper handling of array types)
    unlabeled_set.targets = train_set.targets.copy() if isinstance(train_set.targets, np.ndarray) else train_set.targets.copy()

def _report_split_statistics(args, unlabeled_set, test_set):
    """
    Report statistics about the dataset splits.
    
    Args:
        args: Arguments containing dataset configuration
        unlabeled_set: Unlabeled dataset
        test_set: Testing dataset
    """
    # Split Check and Reporting
    print("Target classes: ", args.target_list)

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
