from torch.utils.data import Dataset
import numpy as np

class MyTinyImageNet(Dataset):
    def __init__(self, hf_dataset, transform=None, imbalance_factor=None, method=None, n_class=200):
        """
        Initialize the TinyImageNet dataset.
        
        Args:
        hf_dataset (Dataset): Hugging Face dataset object.
        transform (callable, optional): Function to preprocess the images.
        imbalance_factor (float, optional): Imbalance factor for the dataset.
        method (str, optional): Name of the method to be used.
        n_class (int): Number of classes, default is 200.
        """
        self.data = hf_dataset['image']  # Changed from 'img' to 'image'
        self.targets = np.array(hf_dataset['label'])  # This is correct
        self.transform = transform
        self.classes = hf_dataset.features['label'].names  # This is correct
        self.method = method
        
        if imbalance_factor:
            # Create imbalance ratios
            imbalance_ratios = np.logspace(np.log10(imbalance_factor), 0, num=n_class)[::-1]
            
            # Get indices for each class
            train_idx_per_class = [np.where(self.targets == i)[0] for i in range(n_class)]
            
            # Resample based on indices
            new_indices = []
            for class_idx, class_indices in enumerate(train_idx_per_class):
                n_samples = int(len(class_indices) * imbalance_ratios[class_idx])
                new_indices.extend(np.random.choice(class_indices, n_samples, replace=False))
            
            # Create imbalanced training dataset
            self.data = [self.data[i] for i in new_indices]
            self.targets = self.targets[new_indices]

    def __getitem__(self, index):
        """
        Get a sample by index.

        Args:
        index (int): Index of the sample.

        Returns:
        tuple: (image, label, index, [moving probability])
        """
        img = self.data[index]
        target = self.targets[index]

        if self.transform:
            img = self.transform(img)

        if hasattr(self, 'moving_prob'):
            moving_prob = self.moving_prob[index]
            return img, target, index, moving_prob

        return img, target, index

    def init_tidal_params(self, n_class):
        """在知道实际类别数后初始化TIDAL参数"""
        if self.method == 'TIDAL':
            self.moving_prob = np.zeros((len(self.data), n_class), dtype=np.float32)

    def __len__(self):
        """
        Return the number of samples in the dataset.

        Returns:
        int: Number of samples.
        """
        return len(self.data)
