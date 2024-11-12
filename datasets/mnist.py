from torchvision import datasets
from torch.utils.data import Dataset
import numpy as np

class MyMNIST(Dataset):
    def __init__(self, file_path, train, download, transform):
        self.mnist = datasets.MNIST(file_path, train=train, download=download, transform=transform)
        self.targets = np.array(self.mnist.targets)
        self.classes = list(range(10))  # MNIST has 10 classes: digits 0-9

    def __getitem__(self, index):
        data, _ = self.mnist[index]
        target = self.targets[index]
        return data, target, index

    def __len__(self):
        return len(self.mnist)
