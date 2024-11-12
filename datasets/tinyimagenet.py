from torchvision import datasets, transforms
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder
from torch import tensor, long
import numpy as np

class MyTinyImageNet(Dataset):
    def __init__(self, file_path, transform=None):
        self.transform = transform
        self.data = ImageFolder(file_path)
        self.targets = np.array(self.data.targets)
        self.classes = self.data.classes

    def __getitem__(self, index):
        img, _ = self.data[index]
        target = self.targets[index]

        if self.transform is not None:
            img = self.transform(img)

        return img, target, index

    def __len__(self):
        return len(self.data)

# Update augmentation function for 64x64 images
def get_augmentations_64(T_normalize):
    train_transform = T.Compose([T.RandomHorizontalFlip(), T.RandomCrop(size=64, padding=4), T.ToTensor(), T_normalize])
    test_transform = T.Compose([T.ToTensor(), T_normalize])
    return train_transform, test_transform

# TinyImageNet-specific dataset handling
def TinyImageNet(args):
    channel = 3
    im_size = (64, 64)  # TinyImageNet image size
    num_classes = 200   # TinyImageNet has 200 classes
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    T_normalize = T.Normalize(mean, std)

    # Get transforms specific to 64x64 images
    train_transform, test_transform = get_augmentations_64(T_normalize)

    # Load datasets from TinyImageNet paths
    dst_train = MyTinyImageNet(args.data_path+'/tiny-imagenet-200/train/', transform=train_transform)
    dst_test = MyTinyImageNet(args.data_path+'/tiny-imagenet-200/val/', transform=test_transform)

    # Get class names from the dataset
    class_names = dst_train.classes

    # Convert targets to tensors
    dst_train.targets = tensor(dst_train.targets, dtype=long)
    dst_test.targets = tensor(dst_test.targets, dtype=long)

    return channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test
