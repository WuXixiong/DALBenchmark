import numpy as np
import torch
from torch.utils.data.dataset import Subset

def get_subset_with_len(dataset, length, shuffle=False):
    """
    Get a subset of the dataset with a specific length.
    
    Args:
        dataset: The original dataset
        length: The desired length of the subset
        shuffle: Whether to shuffle the indices before creating the subset
        
    Returns:
        A subset of the original dataset with the specified length
    """
    dataset_size = len(dataset)

    index = np.arange(dataset_size)
    if shuffle:
        np.random.shuffle(index)

    index = torch.from_numpy(index[0:length])
    subset = Subset(dataset, index)

    assert len(subset) == length

    return subset

def get_balanced_subset_indices(targets, samples_per_class, random_seed=42):
    """
    Get indices for a balanced subset from the dataset.
    
    Args:
        targets: Array of target labels
        samples_per_class: Number of samples to select per class
        random_seed: Random seed for reproducible sampling
        
    Returns:
        List of indices for the balanced subset
    """
    np.random.seed(random_seed)
    
    # Convert to numpy array if not already
    if not isinstance(targets, np.ndarray):
        targets = np.array(targets)
    
    # Get unique classes and their counts
    unique_classes, class_counts = np.unique(targets, return_counts=True)
    
    print(f"Original dataset statistics:")
    for cls, count in zip(unique_classes, class_counts):
        print(f"  Class {cls}: {count} samples")
    
    # Check if we have enough samples per class
    insufficient_classes = []
    for cls, count in zip(unique_classes, class_counts):
        if count < samples_per_class:
            insufficient_classes.append((cls, count))
    
    if insufficient_classes:
        print(f"\nWarning: The following classes have fewer than {samples_per_class} samples:")
        for cls, count in insufficient_classes:
            print(f"  Class {cls}: only {count} samples available")
        print("Will use all available samples for these classes.")
    
    # Sample indices for each class
    selected_indices = []
    actual_samples_per_class = {}
    
    for cls in unique_classes:
        # Get all indices for this class
        class_indices = np.where(targets == cls)[0]
        
        # Determine how many samples to take
        available_samples = len(class_indices)
        samples_to_take = min(samples_per_class, available_samples)
        actual_samples_per_class[cls] = samples_to_take
        
        # Randomly sample indices
        if samples_to_take < available_samples:
            sampled_indices = np.random.choice(class_indices, size=samples_to_take, replace=False)
        else:
            sampled_indices = class_indices
            
        selected_indices.extend(sampled_indices.tolist())
    
    # Shuffle the final indices to avoid class ordering
    np.random.shuffle(selected_indices)
    
    print(f"\nBalanced subset statistics:")
    for cls in unique_classes:
        print(f"  Class {cls}: {actual_samples_per_class[cls]} samples selected")
    print(f"Total samples in subset: {len(selected_indices)}")
    
    return selected_indices

def apply_balanced_subset(dataset, samples_per_class, random_seed=42):
    """
    Apply balanced subset selection to a dataset.
    
    Args:
        dataset: Dataset object with data and targets attributes
        samples_per_class: Number of samples to select per class
        random_seed: Random seed for reproducible sampling
        
    Returns:
        Modified dataset with balanced subset
    """
    if not hasattr(dataset, 'targets'):
        print("Warning: Dataset does not have 'targets' attribute. Skipping subset selection.")
        return dataset
    
    # Convert targets to numpy for processing if it's a torch tensor
    targets_for_processing = dataset.targets
    if hasattr(dataset.targets, 'numpy'):  # PyTorch tensor
        targets_for_processing = dataset.targets.numpy()
    elif hasattr(dataset.targets, 'cpu'):  # PyTorch tensor on GPU
        targets_for_processing = dataset.targets.cpu().numpy()
    
    # Get balanced subset indices
    subset_indices = get_balanced_subset_indices(targets_for_processing, samples_per_class, random_seed)
    
    # Update targets based on type
    if hasattr(dataset.targets, 'numpy') or hasattr(dataset.targets, 'cpu'):  # PyTorch tensor
        import torch
        dataset.targets = dataset.targets[subset_indices]
    elif isinstance(dataset.targets, np.ndarray):
        dataset.targets = dataset.targets[subset_indices]
    elif isinstance(dataset.targets, list):
        dataset.targets = [dataset.targets[i] for i in subset_indices]
    
    # Update data attributes
    if hasattr(dataset, 'data'):
        if isinstance(dataset.data, np.ndarray):
            dataset.data = dataset.data[subset_indices]
        elif isinstance(dataset.data, list):
            dataset.data = [dataset.data[i] for i in subset_indices]
        elif hasattr(dataset.data, 'numpy') or hasattr(dataset.data, 'cpu'):  # PyTorch tensor
            dataset.data = dataset.data[subset_indices]
    
    # Handle other possible data attribute names for text datasets
    for attr_name in ['texts', 'examples', 'inputs', 'sentences', 'reviews']:
        if hasattr(dataset, attr_name):
            attr_value = getattr(dataset, attr_name)
            if isinstance(attr_value, list):
                setattr(dataset, attr_name, [attr_value[i] for i in subset_indices])
            elif isinstance(attr_value, np.ndarray):
                setattr(dataset, attr_name, attr_value[subset_indices])
            elif hasattr(attr_value, 'numpy') or hasattr(attr_value, 'cpu'):  # PyTorch tensor
                setattr(dataset, attr_name, attr_value[subset_indices])
    
    return dataset