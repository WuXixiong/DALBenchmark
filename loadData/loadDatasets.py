import numpy as np
import random
import os
import torch
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
from ownDatasets.trec6 import MyTREC6Dataset
from ownDatasets.rcv1 import MyRCV1Dataset

# Import transforms
from .dataTransform import get_dataset_transforms
from .dataUtils import get_balanced_subset_indices, apply_balanced_subset

def get_dataset(args, trial):
    """
    Load and prepare datasets for training and testing.
    
    Args:
        args: Arguments containing dataset configuration
        trial: Trial number for open set learning
        
    Returns:
        A tuple of (train_set, unlabeled_set, test_set)
    """
    # === Strategy Detection and Configuration ===
    use_balanced_subset = hasattr(args, 'samples_per_class') and args.samples_per_class is not None
    use_imbalance_factor = hasattr(args, 'imb_factor') and args.imb_factor != 1.0
    
    # Save original imbalance factor
    original_imb_factor = getattr(args, 'imb_factor', None) or 1.0
    dataset_strategy = getattr(args, 'dataset_strategy', 'balanced_first')  # Default strategy
    
    # Strategy selection logic
    if use_balanced_subset and use_imbalance_factor:
        print(f"\n=== Detected both balanced subset and imbalance factor settings ===")
        print(f"Balanced subset: {args.samples_per_class} samples per class")
        print(f"Imbalance factor: {args.imb_factor}")
        print(f"Strategy: {dataset_strategy}")
        
        if dataset_strategy == 'balanced_first':
            print("Strategy execution: First create balanced subset, then apply imbalance factor")
            # Temporarily disable imbalance factor, use balanced data when creating dataset
            args.imb_factor = 1.0
        elif dataset_strategy == 'imbalanced_only':
            print("Strategy execution: Only use imbalance factor, ignore balanced subset setting")
            args.samples_per_class = None
        elif dataset_strategy == 'balanced_only':
            print("Strategy execution: Only use balanced subset, ignore imbalance factor")
            args.imb_factor = 1.0
            original_imb_factor = 1.0  # Ensure no recovery later
        else:
            raise ValueError(f"Unknown dataset strategy: {dataset_strategy}")
        print("=" * 50)
    
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
        elif args.dataset == 'TREC6':
            trec6_dataset = load_dataset("trec", trust_remote_code=True)
            train_set = MyTREC6Dataset(trec6_dataset['train'], tokenizer=tokenizer, imbalance_factor=args.imb_factor)
            test_set = MyTREC6Dataset(trec6_dataset['test'], tokenizer=tokenizer, imbalance_factor=args.imb_factor)
            unlabeled_set = MyTREC6Dataset(trec6_dataset['train'], tokenizer=tokenizer, imbalance_factor=args.imb_factor)
        elif args.dataset == 'RCV1':
            # RCV1 dataset paths 
            data_dir = getattr(args, 'data_dir', '/home/cwu1/sv12/chenkaiwu/DALBenchmark/data/rcv1/')
            train_file = os.path.join(data_dir, 'rcv1_train_meanir_10.json')
            test_file = os.path.join(data_dir, 'rcv1_test_meanir_10.json')
            
            # RCV1 parameters
            max_length = getattr(args, 'max_length', 512)  # RCV1 documents are usually longer
            use_keywords = getattr(args, 'use_keywords', False)  # Whether to use keywords
            
            # Create training dataset
            train_set = MyRCV1Dataset(
                data_files=train_file, 
                tokenizer=tokenizer, 
                imbalance_factor=args.imb_factor,
                max_length=max_length,
                use_keywords=use_keywords
            )
            
            # Create test dataset  
            test_set = MyRCV1Dataset(
                data_files=test_file, 
                tokenizer=tokenizer, 
                imbalance_factor=args.imb_factor, 
                max_length=max_length,
                use_keywords=use_keywords
            )
            
            test_set.mlb = train_set.mlb
            test_set.classes = train_set.classes  
            test_set.num_classes = train_set.num_classes
            test_set.encoded_labels = test_set.mlb.transform(test_set.labels)
            test_set.targets = torch.tensor(test_set.encoded_labels, dtype=torch.float)
            
            unlabeled_set = MyRCV1Dataset(
                data_files=train_file, 
                tokenizer=tokenizer, 
                imbalance_factor=args.imb_factor,
                max_length=max_length,
                use_keywords=use_keywords
            )
        else:
            raise ValueError(f"Text dataset '{args.dataset}' is not supported. Please choose from the available text datasets.")
    else:
        raise ValueError(f"Dataset '{args.dataset}' is not supported. Please choose from the available datasets.")

    # === Apply balanced subset selection with strategy control ===
    if hasattr(args, 'samples_per_class') and args.samples_per_class is not None:
        print(f"\n=== Applying balanced subset selection ===")
        print(f"Target samples per class: {args.samples_per_class}")
        
        # Apply to training set
        print(f"\nProcessing training set:")
        train_set = apply_balanced_subset(train_set, args.samples_per_class, 
                                        getattr(args, 'subset_random_seed', 42))
        
        # Apply to test set if specified - maintain same distribution as training set
        if hasattr(args, 'apply_subset_to_test') and args.apply_subset_to_test:
            print(f"\nProcessing test set:")
            test_samples_per_class = int(args.samples_per_class / 10)
            test_set = apply_balanced_subset(test_set, test_samples_per_class, 
                                           getattr(args, 'subset_random_seed', 42) + 1)
        
        # Update unlabeled set to match training set
        _update_unlabeled_set(train_set, unlabeled_set)
        
        # === Apply imbalance factor according to strategy ===
        if dataset_strategy == 'balanced_first' and original_imb_factor != 1.0:
            print(f"\n=== Applying imbalance factor on balanced subset ===")
            print(f"Imbalance factor: {original_imb_factor}")
            
            # Apply imbalance on training set balanced subset
            train_set = apply_imbalance_to_balanced_subset(train_set, original_imb_factor, 
                                                         getattr(args, 'subset_random_seed', 42) + 10)
            
            # Apply SAME imbalance factor to test set if test subset was created
            if hasattr(args, 'apply_subset_to_test') and args.apply_subset_to_test:
                print(f"\nApplying same imbalance factor to test set:")
                test_set = apply_imbalance_to_balanced_subset(test_set, original_imb_factor, 
                                                            getattr(args, 'subset_random_seed', 42) + 20)
            
            # Update unlabeled set to match training set
            _update_unlabeled_set(train_set, unlabeled_set)
            
            # Restore original imbalance factor to args (for other potential uses)
            args.imb_factor = original_imb_factor
            
            print(f"=== Imbalance factor application completed ===")
        
        print(f"=== Balanced subset selection completed ===\n")
    
    # Configure dataset settings based on type
    _configure_dataset_settings(args, trial)

    # === Ensure test-train distribution consistency ===
    # This is crucial for realistic evaluation under imbalanced conditions
    if (hasattr(args, 'apply_subset_to_test') and args.apply_subset_to_test and 
        hasattr(args, 'imb_factor') and args.imb_factor != 1.0 and
        dataset_strategy == 'imbalanced_only'):
        # For imbalanced_only strategy, ensure test set follows same pattern
        print(f"\n=== Ensuring test-train distribution consistency ===")
        test_set = ensure_test_train_distribution_consistency(args, train_set, test_set)
        print(f"=== Test-train distribution consistency ensured ===\n")

    # Apply class conversion for different dataset types
    _apply_class_conversion(args, train_set, test_set, unlabeled_set)
    
    # Report split statistics
    _report_split_statistics(args, unlabeled_set, test_set)
    
    return train_set, unlabeled_set, test_set


def apply_imbalance_to_balanced_subset(dataset, imbalance_factor, random_seed=42):
    """
    Apply imbalance factor to an existing balanced dataset
    
    Args:
        dataset: Balanced dataset
        imbalance_factor: Imbalance factor (0.01-1.0, smaller values mean more imbalanced)
        random_seed: Random seed
        
    Returns:
        Dataset with applied imbalance
    """
    if imbalance_factor == 1.0:
        print("Imbalance factor is 1.0, maintaining balanced distribution")
        return dataset
    
    np.random.seed(random_seed)
    
    # Get all labels
    if hasattr(dataset, 'targets'):
        if isinstance(dataset.targets, torch.Tensor):
            targets = dataset.targets.numpy()
        else:
            targets = np.array(dataset.targets)
    else:
        targets = np.array([dataset[i]['labels'] for i in range(len(dataset))])
    
    unique_labels = np.unique(targets)
    num_classes = len(unique_labels)
    
    print(f"Applying imbalance factor {imbalance_factor} to {num_classes} classes")
    
    # Create class sample index mapping
    class_indices = {label: np.where(targets == label)[0] for label in unique_labels}
    
    # Get base sample count (current minimum class sample count, since it's balanced, all are equal)
    base_count = len(class_indices[unique_labels[0]])
    print(f"Balanced subset samples per class: {base_count}")
    
    selected_indices = []
    class_sample_counts = {}
    
    for i, label in enumerate(sorted(unique_labels)):
        indices = class_indices[label]
        
        # Calculate how many samples this class should keep - use exponential decay to create long-tail distribution
        if num_classes == 1:
            keep_count = base_count
        else:
            # Normalize position to [0, 1]
            position = i / (num_classes - 1)
            # Apply exponential decay
            keep_count = int(base_count * (imbalance_factor ** position))
        
        # Ensure at least some samples are kept
        if imbalance_factor >= 0.01:
            keep_count = max(1, keep_count)
        
        keep_count = min(keep_count, len(indices))
        
        # Randomly select samples to keep
        if keep_count > 0:
            selected = np.random.choice(indices, keep_count, replace=False)
            selected_indices.extend(selected)
            class_sample_counts[label] = keep_count
        else:
            class_sample_counts[label] = 0
    
    # Print imbalanced distribution information
    print("Imbalanced distribution:")
    for label in sorted(unique_labels):
        count = class_sample_counts.get(label, 0)
        print(f"  Class {label}: {count} samples")
    
    # Update dataset
    selected_indices = sorted(selected_indices)
    _update_dataset_with_indices(dataset, selected_indices)
    
    print(f"Total {len(selected_indices)} samples retained")
    
    return dataset


def ensure_test_train_distribution_consistency(args, train_set, test_set):
    """
    Ensure test set follows the same distribution pattern as training set
    This is crucial for realistic evaluation under imbalanced conditions
    
    Args:
        args: Arguments containing dataset configuration
        train_set: Training dataset (already processed)
        test_set: Test dataset (to be processed to match train distribution)
        
    Returns:
        test_set: Test dataset with matching distribution
    """
    if not (hasattr(args, 'apply_subset_to_test') and args.apply_subset_to_test):
        print("Test set processing disabled, using original test set distribution")
        return test_set
    
    # Get training set distribution for reference
    if hasattr(train_set, 'targets'):
        if isinstance(train_set.targets, torch.Tensor):
            train_targets = train_set.targets.numpy()
        else:
            train_targets = np.array(train_set.targets)
    else:
        train_targets = np.array([train_set[i]['labels'] for i in range(len(train_set))])
    
    train_unique, train_counts = np.unique(train_targets, return_counts=True)
    train_distribution = dict(zip(train_unique, train_counts))
    
    print(f"\nEnsuring test set matches training set distribution pattern:")
    print(f"Training set distribution: {train_distribution}")
    
    # Calculate target test set distribution (scaled down proportionally)
    total_train_samples = sum(train_counts)
    target_test_total = int(total_train_samples * 0.2)  # Typical test set is ~20% of train set
    
    test_distribution = {}
    for label, count in train_distribution.items():
        proportion = count / total_train_samples
        test_distribution[label] = max(1, int(target_test_total * proportion))
    
    print(f"Target test set distribution: {test_distribution}")
    
    # Apply this distribution to test set
    test_set = apply_specific_distribution(test_set, test_distribution, 
                                         getattr(args, 'subset_random_seed', 42) + 30)
    
    return test_set


def apply_specific_distribution(dataset, target_distribution, random_seed=42):
    """
    Apply a specific class distribution to a dataset
    
    Args:
        dataset: Dataset to modify
        target_distribution: Dict mapping class labels to desired sample counts
        random_seed: Random seed
        
    Returns:
        Modified dataset
    """
    np.random.seed(random_seed)
    
    # Get current dataset targets
    if hasattr(dataset, 'targets'):
        if isinstance(dataset.targets, torch.Tensor):
            targets = dataset.targets.numpy()
        else:
            targets = np.array(dataset.targets)
    else:
        targets = np.array([dataset[i]['labels'] for i in range(len(dataset))])
    
    # Create class sample index mapping
    class_indices = {}
    for label in target_distribution.keys():
        class_indices[label] = np.where(targets == label)[0]
    
    selected_indices = []
    actual_distribution = {}
    
    for label, target_count in target_distribution.items():
        available_indices = class_indices.get(label, [])
        actual_count = min(target_count, len(available_indices))
        
        if actual_count > 0:
            selected = np.random.choice(available_indices, actual_count, replace=False)
            selected_indices.extend(selected)
            actual_distribution[label] = actual_count
        else:
            actual_distribution[label] = 0
    
    print(f"Actual applied distribution: {actual_distribution}")
    
    # Update dataset
    selected_indices = sorted(selected_indices)
    _update_dataset_with_indices(dataset, selected_indices)
    
    return dataset


def _update_dataset_with_indices(dataset, selected_indices):
    """
    Update dataset using selected indices
    
    Args:
        dataset: Dataset to update
        selected_indices: List of selected sample indices
    """
    # Update different types of data attributes
    if hasattr(dataset, 'data'):
        if isinstance(dataset.data, list):
            dataset.data = [dataset.data[i] for i in selected_indices]
        elif isinstance(dataset.data, np.ndarray):
            dataset.data = dataset.data[selected_indices]
        elif hasattr(dataset.data, '__getitem__'):  # Handle other indexable types
            dataset.data = [dataset.data[i] for i in selected_indices]
    
    if hasattr(dataset, 'targets'):
        if isinstance(dataset.targets, torch.Tensor):
            dataset.targets = dataset.targets[selected_indices]
        elif isinstance(dataset.targets, list):
            dataset.targets = [dataset.targets[i] for i in selected_indices]
        elif isinstance(dataset.targets, np.ndarray):
            dataset.targets = dataset.targets[selected_indices]
    
    # Handle special attributes for text datasets
    if hasattr(dataset, 'texts'):
        dataset.texts = [dataset.texts[i] for i in selected_indices]
    
    if hasattr(dataset, 'labels'):
        if isinstance(dataset.labels, list):
            dataset.labels = [dataset.labels[i] for i in selected_indices]
        else:
            dataset.labels = dataset.labels[selected_indices]
    
    if hasattr(dataset, 'examples'):
        dataset.examples = [dataset.examples[i] for i in selected_indices]


def _update_unlabeled_set(train_set, unlabeled_set):
    """Update unlabeled dataset to match training set"""
    if hasattr(train_set, 'data'):
        unlabeled_set.data = train_set.data.copy() if hasattr(train_set.data, 'copy') else train_set.data[:]
    elif hasattr(train_set, 'texts'):
        unlabeled_set.texts = train_set.texts.copy()
        unlabeled_set.labels = train_set.labels.copy()
    elif hasattr(train_set, 'examples'):
        unlabeled_set.examples = train_set.examples.copy()
    
    unlabeled_set.targets = train_set.targets.copy() if hasattr(train_set.targets, 'copy') else train_set.targets[:]


# === The following are original helper functions, kept unchanged ===

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
            if args.ood_rate == 0.6:
                args.target_lists = [[4, 2, 5, 7], [7, 1, 2, 5], [6, 4, 3, 2], [8, 9, 1, 3], [2, 9, 5, 3], [3, 6, 4, 7]]
            if args.ood_rate == 0.4:
                args.target_lists = [[1, 3, 4, 2, 5, 7], [6, 9, 7, 1, 2, 5], [5, 1, 6, 4, 3, 2], [7, 2, 8, 9, 1, 3], [8, 1, 2, 9, 5, 3], [8, 5, 3, 6, 4, 7]]
            if args.ood_rate == 0.8:
                args.target_lists = [[4, 2], [7, 1], [6, 4], [8, 9], [2, 9], [3, 6]]
            if args.ood_rate == 0.2:
                args.target_lists = [[1, 3, 4, 2, 5, 7, 6, 8], [0, 3, 6, 9, 7, 1, 2, 5], [8, 9, 5, 1, 6, 4, 3, 2], [5, 6, 7, 2, 8, 9, 1, 3], [4, 6, 8, 1, 2, 9, 5, 3], [0, 2, 8, 5, 3, 6, 4, 7]]
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
    
    elif args.dataset in ['SST5', 'YELP']:
        args.input_size = 128
        # Set total number of classes (including OOD)
        args.n_class = 5  # Total number of classes in SST5
        args.target_list = list(range(5))  # All classes (for closed set)
        args.num_IN_class = 5  # Number of in-distribution classes
        if args.openset:
            # Define different class combinations for trials
            if args.ood_rate == 0.6:
                args.target_lists = [
                    [0, 2], [1, 3], [2, 4], [3, 4],
                    [1, 4], [0, 3], [1, 2],
                    [2, 3], [0, 4],
                    [3, 4]
                ]
            if args.ood_rate == 0.4:
                args.target_lists = [
                    [0, 2 ,3], [1, 3, 4], [2, 4 ,0], [3, 4, 0],
                    [1, 4, 0]
                ]
            if args.ood_rate == 0.2:
                args.target_lists = [
                    [0, 1, 2 ,3], [1, 2, 3, 4], [1, 2, 4 ,0], [1, 3, 4, 0],
                    [2, 3, 4, 0]
                ]
            if trial >= len(args.target_lists):
                print(f"Warning: Trial {trial} exceeds available target lists. Using trial % len(target_lists).")
            args.target_list = args.target_lists[trial % len(args.target_lists)]
            args.num_IN_class = len(args.target_list)  # Update number of in-distribution classes
        
        # Calculate untarget_list (classes that will be treated as OOD)
        args.untarget_list = list(np.setdiff1d(list(range(args.n_class)), list(args.target_list)))

    elif args.dataset == 'RCV1':
        args.input_size = 512  # RCV1 documents are usually long, so a larger input size is needed

        # The number of classes in RCV1 should be determined based on the actual data
        # Here we set a placeholder value
        if not hasattr(args, 'n_class'):
            # If the dataset has not been loaded yet, set a default value
            # The actual value will be updated after creating the dataset
            args.n_class = 103  # Common number of topic classes in RCV1; adjust based on your data

        # For multi-label classification, target_list includes all possible labels
        args.target_list = list(range(args.n_class))  # All possible labels
        args.num_IN_class = args.n_class  # For multi-label tasks, all classes are in-distribution

        # RCV1-specific configurations
        args.is_multilabel = True  # Indicate this is a multi-label task
        args.loss_type = 'bce'  # BCE loss is commonly used for multi-label classification

        # If class imbalance needs to be addressed, set class balancing parameters
        args.class_balanced = getattr(args, 'class_balanced', False)

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

    elif args.dataset == 'TREC6':
        args.input_size = 128
        # Set total number of classes (including OOD)
        args.n_class = 6  # Total number of classes in TREC6
        args.target_list = list(range(6))  # All classes (for closed set)
        args.num_IN_class = 6  # Number of in-distribution classes
        
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
        args.n_class = 14  # Total number of classes in DBPEDIA
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
            if args.ood_rate == 0.6:
            # Settings for open set learning (compact representation of the original lists)
                args.target_lists = [
                [69, 8, 86, 18, 68, 30, 75, 3, 63, 76, 72, 7, 50, 81, 46, 89, 22, 93, 62, 21, 33, 98, 82, 20, 60, 5, 77, 1, 74, 88, 57, 34, 43, 27, 66, 83, 25, 48, 4, 55], 
                [33, 10, 74, 72, 88, 47, 27, 68, 60, 75, 45, 79, 92, 35, 86, 50, 18, 61, 49, 29, 23, 30, 67, 73, 82, 94, 13, 37, 39, 26, 62, 22, 90, 53, 89, 11, 3, 20, 70, 96], 
                [70, 28, 60, 22, 39, 35, 73, 13, 74, 10, 2, 16, 80, 53, 67, 66, 78, 46, 26, 71, 43, 38, 42, 14, 50, 77, 20, 48, 52, 8, 54, 58, 91, 5, 25, 90, 61, 11, 59, 55], 
                [7, 93, 37, 84, 57, 99, 10, 75, 54, 42, 26, 27, 47, 52, 61, 86, 60, 90, 1, 0, 98, 87, 94, 74, 56, 91, 23, 97, 30, 17, 53, 12, 76, 11, 25, 65, 96, 3, 45, 8], 
                [0, 1, 4, 5, 7, 9, 12, 19, 21, 22, 23, 24, 38, 41, 42, 43, 46, 47, 48, 51, 55, 59, 60, 62, 68, 73, 75, 78, 79, 80, 81, 85, 86, 90, 91, 94, 95, 96, 97, 98]
            ]

            if args.ood_rate == 0.4:
                args.target_lists = [
                [0, 23, 62, 78, 38, 89, 41, 22, 35, 65, 48, 74, 66, 50, 25, 92, 49, 19, 30, 75, 98, 26, 81, 34, 84, 93, 69, 10, 14, 32, 97, 91, 27, 80, 86, 36, 45, 87, 16, 85, 43, 94, 70, 9, 12, 11, 40, 60, 71, 59, 58, 3, 52, 15, 77, 21, 8, 79, 64, 20],
                [66, 9, 33, 86, 1, 50, 98, 25, 71, 99, 84, 20, 89, 28, 97, 93, 10, 41, 57, 59, 75, 2, 73, 45, 85, 38, 95, 8, 64, 94, 11, 16, 3, 32, 48, 61, 82, 65, 13, 56, 23, 36, 40, 81, 96, 35, 46, 72, 42, 12, 92, 77, 21, 17, 87, 7, 62, 68, 69, 74],
                [42, 61, 68, 0, 1, 79, 84, 24, 98, 3, 45, 29, 40, 10, 25, 85, 87, 67, 39, 69, 14, 54, 41, 76, 95, 48, 82, 88, 52, 35, 8, 5, 17, 70, 51, 77, 43, 22, 50, 19, 90, 92, 34, 37, 11, 91, 62, 86, 64, 23, 63, 83, 13, 58, 55, 78, 99, 30, 32, 15],
                [32, 66, 89, 81, 95, 53, 58, 68, 71, 8, 82, 50, 39, 72, 43, 60, 51, 57, 28, 14, 98, 19, 20, 18, 86, 59, 73, 37, 41, 33, 67, 54, 26, 65, 24, 17, 90, 27, 87, 56, 16, 77, 12, 25, 34, 93, 38, 79, 94, 88, 91, 85, 78, 13, 84, 6, 83, 42, 4, 5],
                [11, 1, 33, 4, 35, 0, 13, 8, 87, 29, 34, 21, 45, 43, 83, 81, 79, 65, 66, 3, 46, 22, 99, 89, 36, 80, 78, 94, 23, 14, 90, 86, 62, 10, 72, 42, 12, 85, 49, 69, 37, 53, 7, 40, 64, 98, 70, 56, 6, 88, 26, 47, 16, 73, 27, 61, 48, 68, 97, 58]
                ]
            
            if args.ood_rate == 0.2:
                args.target_lists = [
                [19, 45, 49, 89, 85, 3, 97, 93, 34, 59, 47, 80, 4, 50, 25, 74, 86, 61, 7, 17, 48, 58, 30, 96, 79, 99, 13, 62, 27, 14, 53, 16, 32, 23, 40, 98, 0, 24, 65, 31, 72, 90, 92, 51, 15, 60, 38, 11, 41, 82, 36, 67, 84, 28, 8, 54, 94, 43, 18, 5, 78, 6, 29, 1, 73, 35, 76, 22, 69, 37, 91, 42, 75, 56, 44, 87, 83, 9, 39, 77],
                [3, 96, 2, 68, 21, 73, 64, 94, 77, 23, 97, 88, 53, 41, 67, 11, 42, 27, 38, 48, 99, 59, 37, 80, 82, 71, 19, 52, 10, 28, 14, 87, 47, 58, 16, 55, 98, 5, 49, 34, 6, 29, 24, 61, 75, 51, 26, 45, 17, 43, 78, 56, 81, 50, 72, 74, 79, 18, 84, 33, 22, 4, 9, 12, 32, 93, 39, 0, 95, 63, 25, 62, 76, 91, 31, 20, 30, 85, 60, 54],
                [30, 8, 49, 18, 64, 73, 10, 1, 46, 29, 72, 75, 95, 99, 35, 57, 68, 36, 76, 21, 96, 54, 90, 65, 6, 89, 25, 74, 63, 55, 98, 62, 58, 37, 51, 78, 43, 14, 56, 81, 48, 47, 70, 82, 85, 87, 94, 41, 13, 92, 66, 34, 33, 97, 3, 71, 91, 67, 32, 60, 40, 52, 79, 69, 77, 53, 86, 15, 7, 19, 93, 17, 16, 39, 2, 50, 88, 5, 4, 26],
                [20, 35, 19, 12, 32, 45, 0, 24, 55, 89, 2, 5, 82, 46, 49, 86, 71, 16, 11, 90, 66, 50, 15, 99, 74, 51, 57, 61, 65, 3, 43, 21, 37, 53, 91, 7, 76, 54, 36, 92, 29, 41, 33, 95, 63, 67, 23, 30, 9, 87, 6, 22, 31, 96, 34, 14, 93, 85, 98, 28, 69, 84, 94, 25, 81, 70, 72, 59, 78, 77, 73, 48, 26, 88, 60, 13, 79, 42, 10, 38],
                [1, 70, 92, 45, 9, 7, 46, 17, 39, 93, 33, 49, 6, 58, 89, 16, 48, 43, 52, 50, 74, 98, 99, 36, 55, 69, 37, 3, 84, 19, 59, 83, 12, 35, 91, 66, 85, 54, 75, 22, 68, 13, 76, 29, 11, 57, 73, 77, 60, 41, 95, 64, 47, 40, 62, 80, 25, 96, 90, 97, 18, 72, 44, 65, 86, 78, 81, 31, 87, 4, 88, 56, 28, 71, 21, 5, 51, 10, 26, 15]
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
    is_multilabel = getattr(args, 'is_multilabel', False)
    
    if is_multilabel:
        # For multi-label tasks, targets are typically already in the correct format
        # Multi-label datasets usually don't need class remapping
        print("Multi-label dataset: skipping class conversion (targets assumed to be in correct format)")
        
        # Ensure targets are numpy arrays
        if hasattr(train_set, 'targets'):
            if not isinstance(train_set.targets, np.ndarray):
                train_set.targets = np.array(train_set.targets)
        if hasattr(test_set, 'targets'):
            if not isinstance(test_set.targets, np.ndarray):
                test_set.targets = np.array(test_set.targets)
        
        # Copy train_set targets to unlabeled_set
        if hasattr(train_set, 'targets'):
            unlabeled_set.targets = train_set.targets.copy()
        
        return

    # Single-label processing (original logic)
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
    is_multilabel = getattr(args, 'is_multilabel', False)
    
    # Split Check and Reporting
    print("Target classes: ", args.target_list)
         
    # Add subset selection info if applied
    if hasattr(args, 'samples_per_class') and args.samples_per_class is not None:
        print(f"Balanced subset applied: {args.samples_per_class} samples per class")
        if hasattr(args, 'dataset_strategy'):
            print(f"Dataset strategy: {args.dataset_strategy}")
        if hasattr(args, 'imb_factor') and args.imb_factor != 1.0:
            print(f"Final imbalance factor: {args.imb_factor}")
        if hasattr(args, 'apply_subset_to_test') and args.apply_subset_to_test:
            print(f"Test set processing: Same strategy applied (maintains distribution consistency)")

    def count_labels(dataset_targets, dataset_name):
        """Helper function to count labels for both single-label and multi-label"""
        targets = np.array(dataset_targets)
        
        if is_multilabel:
            # Multi-label case
            if len(targets.shape) == 2:
                # Multi-hot encoded matrix
                label_counts = targets.sum(axis=0)
            else:
                # Handle other multi-label formats
                label_counts = np.zeros(len(args.target_list))
                for target in targets:
                    if isinstance(target, (torch.Tensor, np.ndarray)):
                        if target.dim() > 0 and target.shape[0] > 1:
                            active_labels = torch.nonzero(target, as_tuple=True)[0].cpu().numpy() if isinstance(target, torch.Tensor) else np.nonzero(target)[0]
                            label_counts[active_labels] += 1
                        else:
                            label_idx = target.item() if hasattr(target, 'item') else target
                            if 0 <= label_idx < len(args.target_list):
                                label_counts[label_idx] += 1
                    elif isinstance(target, (list, tuple)):
                        for label_idx in target:
                            if 0 <= label_idx < len(args.target_list):
                                label_counts[label_idx] += 1
                    else:
                        if 0 <= target < len(args.target_list):
                            label_counts[target] += 1
            
            uni = np.arange(len(args.target_list))
            cnt = label_counts.astype(int)
        else:
            # Single-label case (original logic)
            uni, cnt = np.unique(targets, return_counts=True)
        
        print(f"{dataset_name}, # samples per class")
        print(uni, cnt)
        
        # Calculate and display imbalance ratio for better understanding
        if len(cnt) > 1:
            max_count = np.max(cnt)
            min_count = np.min(cnt)
            imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')
            print(f"{dataset_name} imbalance ratio (max/min): {imbalance_ratio:.2f}")

    if args.method == 'EPIG':
        targets = np.array(unlabeled_set.targets)
        if not is_multilabel:
            uni, cnt = np.unique(targets, return_counts=True)
            cnt -= args.target_per_class
            print("Train, # samples per class")
            print(uni, cnt)
        else:
            count_labels(unlabeled_set.targets, "Train")
    else:
        count_labels(unlabeled_set.targets, "Train")
         
    count_labels(test_set.targets, "Test")


# === Configuration helper functions ===
def setup_dataset_strategy(args):
    """
    Helper function to setup dataset strategy
    
    Args:
        args: Parameter object
    
    Returns:
        Configured parameter object
    """
    # Strategy options:
    # 'balanced_first': First create balanced subset, then apply imbalance factor (recommended)
    # 'imbalanced_only': Only use imbalance factor, ignore balanced subset
    # 'balanced_only': Only use balanced subset, ignore imbalance factor
    
    if not hasattr(args, 'dataset_strategy'):
        args.dataset_strategy = 'balanced_first'  # Default strategy
    
    valid_strategies = ['balanced_first', 'imbalanced_only', 'balanced_only']
    if args.dataset_strategy not in valid_strategies:
        raise ValueError(f"Unsupported dataset strategy: {args.dataset_strategy}. Available strategies: {valid_strategies}")
    
    print(f"Dataset strategy set: {args.dataset_strategy}")
    
    return args