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
