import random
import torch
import numpy as np
from torch.utils.data.dataset import Subset

def get_subclass_dataset(dataset, classes):
    """
    Extract a subset of the dataset containing only the specified classes.
    
    Args:
        dataset: The original dataset
        classes: A list of class indices or a single class index
        
    Returns:
        A subset of the original dataset containing only the specified classes
    """
    if not isinstance(classes, list):
        classes = [classes]

    indices = []
    for idx, tgt in enumerate(dataset.targets):
        if tgt in classes:
            indices.append(idx)

    dataset = Subset(dataset, indices)
    return dataset

def is_in_distribution_sample(labels, num_classes, is_multilabel=False):
    """
    Check if a sample is in-distribution based on its labels.
    
    Args:
        labels: Label(s) for the sample
        num_classes: Number of in-distribution classes
        is_multilabel: Whether this is a multi-label task
        
    Returns:
        True if sample is in-distribution, False otherwise
    """
    if is_multilabel:
        # For multi-label: check if sample has any valid labels
        if isinstance(labels, (torch.Tensor, np.ndarray)):
            # If labels is a multi-hot vector, check if any positive labels are within range
            if len(labels.shape) > 0 and labels.shape[0] == num_classes:
                return torch.any(labels > 0) if isinstance(labels, torch.Tensor) else np.any(labels > 0)
            else:
                # If labels is a list of active label indices
                if isinstance(labels, torch.Tensor):
                    labels = labels.cpu().numpy() if labels.is_cuda else labels.numpy()
                return any(label < num_classes for label in labels)
        elif isinstance(labels, list):
            # If labels is a list of active label indices
            return any(label < num_classes for label in labels)
        else:
            # Single label case treated as list
            return labels < num_classes
    else:
        # For single-label: simple comparison
        if isinstance(labels, (torch.Tensor, np.ndarray)):
            labels = labels.item() if hasattr(labels, 'item') else labels
        return labels < num_classes

def get_sub_train_dataset(args, dataset, L_index, O_index, U_index, Q_index=None, initial=False):
    """
    Get subsets of the training dataset for active learning.
    
    Args:
        args: Arguments containing dataset configuration
        dataset: The dataset to sample from
        L_index: Indices of labeled samples
        O_index: Indices of out-of-distribution samples
        U_index: Indices of unlabeled samples
        Q_index: Indices of queried samples (used when initial=False)
        initial: Whether this is the initial sampling or an update after querying
        
    Returns:
        If initial=True: A tuple of (L_index, O_index, U_index)
        If initial=False: A tuple of (L_index, O_index, U_index, num_in_query)
    """
    classes = args.target_list
    budget = args.n_initial
    ood_rate = args.ood_rate
    is_multilabel = getattr(args, 'is_multilabel', False)

    if initial:
        if args.openset:
            # Handle text datasets differently
            if args.textset:
                L_total = []
                O_total = []
                for i in range(len(dataset)):
                    sample_labels = dataset[i]['labels']
                    sample_index = dataset[i]['index']
                    
                    if is_in_distribution_sample(sample_labels, len(classes), is_multilabel):
                        L_total.append(sample_index)
                    else:
                        O_total.append(sample_index)
            else:
                # Handle image datasets
                L_total = []
                O_total = []
                for i in range(len(dataset)):
                    sample_labels = dataset[i][1]
                    sample_index = dataset[i][2]
                    
                    if is_in_distribution_sample(sample_labels, len(classes), is_multilabel):
                        L_total.append(sample_index)
                    else:
                        O_total.append(sample_index)

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
            if args.textset:
                L_total = []
                for i in range(len(dataset)):
                    sample_labels = dataset[i]['labels']
                    sample_index = dataset[i]['index']
                    
                    if is_in_distribution_sample(sample_labels, len(classes), is_multilabel):
                        L_total.append(sample_index)
            else:
                L_total = []
                for i in range(len(dataset)):
                    sample_labels = dataset[i][1]
                    sample_index = dataset[i][2]
                    
                    if is_in_distribution_sample(sample_labels, len(classes), is_multilabel):
                        L_total.append(sample_index)
            
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
        if args.textset:
            Q_label = [dataset[i]['labels'] for i in Q_index]
        else:
            Q_label = [dataset[i][1] for i in Q_index]

        # Separate in-distribution and OOD queries
        in_Q_index, ood_Q_index = [], []
        for i, sample_labels in enumerate(Q_label):
            if is_in_distribution_sample(sample_labels, len(classes), is_multilabel):
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
    """
    Get a subset of test dataset containing only in-distribution classes.
    
    Args:
        args: Arguments containing dataset configuration
        dataset: The test dataset
        
    Returns:
        Indices of in-distribution test samples
    """
    classes = args.target_list
    is_multilabel = getattr(args, 'is_multilabel', False)

    labeled_index = []
    if args.textset:
        for i in range(len(dataset)):
            sample_labels = dataset[i]['labels']
            sample_index = dataset[i]['index']
            
            if is_in_distribution_sample(sample_labels, len(classes), is_multilabel):
                labeled_index.append(sample_index)
    else: 
        # for image datasets
        for i in range(len(dataset)):
            sample_labels = dataset[i][1]
            sample_index = dataset[i][2]
            
            if is_in_distribution_sample(sample_labels, len(classes), is_multilabel):
                labeled_index.append(sample_index)
        
    return labeled_index