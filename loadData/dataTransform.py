import torchvision.transforms as T
from torchvision.transforms import Lambda

def get_dataset_transforms(dataset_name):
    """
    Get dataset-specific transforms for training and testing.
    
    Args:
        dataset_name: Name of the dataset (e.g., 'CIFAR10', 'MNIST')
        
    Returns:
        A tuple of (train_transform, test_transform)
    """
    # Normalization for image datasets
    if dataset_name == 'CIFAR10':
        T_normalize = T.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
    elif dataset_name == 'CIFAR100':
        T_normalize = T.Normalize([0.5071, 0.4867, 0.4408], [0.2675, 0.2565, 0.2761])
    elif dataset_name == 'TINYIMAGENET':
        T_normalize = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    elif dataset_name == 'MNIST':
        T_normalize = T.Normalize([0.1307], [0.3081])  # Mean and std for MNIST
    elif dataset_name == 'SVHN':
        T_normalize = T.Normalize([0.4377, 0.4438, 0.4728], [0.1980, 0.2010, 0.1970])  # Mean and std for SVHN
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    # Transform
    if dataset_name in ['CIFAR10', 'CIFAR100', 'SVHN']:
        train_transform = T.Compose([
            T.RandomHorizontalFlip(), 
            T.RandomCrop(size=32, padding=4), 
            T.ToTensor(), 
            T_normalize
        ])
        test_transform = T.Compose([
            T.ToTensor(), 
            T_normalize
        ])
    elif dataset_name == 'TINYIMAGENET':
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
    elif dataset_name == 'MNIST':
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
    
    return train_transform, test_transform
