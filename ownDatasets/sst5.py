from torch.utils.data import Dataset
import torch

class MySST5Dataset(Dataset):
    def __init__(self, hf_dataset, tokenizer, max_length=128):
        """
        初始化 SST-5 数据集。

        参数：
        hf_dataset (Dataset): Hugging Face 加载的数据集对象。
        tokenizer (BertTokenizer): 用于文本编码的分词器。
        max_length (int): 文本序列的最大长度（默认值：128）。
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = hf_dataset['text']
        self.targets = torch.tensor(hf_dataset['label'], dtype=torch.long)
        self.classes = ['very negative', 'negative', 'neutral', 'positive', 'very positive']

    def __getitem__(self, index):
        """
        根据索引获取样本。

        参数：
        index (int): 样本的索引。

        返回：
        dict: 包含 input_ids、attention_mask、labels 和 index 的字典。
        """
        text = self.data[index]
        label = self.targets[index]

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': label,
            'index': index
        }

    def __len__(self):
        """
        返回数据集的样本数量。

        返回：
        int: 样本数量。
        """
        return len(self.data)
