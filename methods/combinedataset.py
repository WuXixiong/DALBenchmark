from torch.utils.data import Dataset, DataLoader
import torch

class PairedDataset(Dataset):
    def __init__(self, labeled_dataset, unlabeled_dataset):
        self.labeled_dataset = labeled_dataset
        self.unlabeled_dataset = unlabeled_dataset
        # 计算两个数据集的最大长度
        self.max_length = max(len(labeled_dataset), len(unlabeled_dataset))

    def __len__(self):
        # 返回两个数据集中的最大长度
        return self.max_length

    def __getitem__(self, index):
        # 循环访问较小的数据集
        labeled_index = index % len(self.labeled_dataset)
        unlabeled_index = index % len(self.unlabeled_dataset)

        # 从标注数据集获取数据和标签
        labeled_data, labeled_label, *_ = self.labeled_dataset[labeled_index]
        # 从未标注数据集获取数据
        unlabeled_data, *_ = self.unlabeled_dataset[unlabeled_index]

        # 返回的格式改为(index, label_x, label_y, unlabel_x, unlabel_y)
        return index, labeled_data, labeled_label, unlabeled_data, torch.tensor(-1)  # 使用-1作为未标注数据的标签
