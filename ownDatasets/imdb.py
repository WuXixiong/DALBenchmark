# from torch.utils.data import Dataset
# from torchtext.datasets import IMDB
# import numpy as np
# import torch

# class MyIMDBDataset(Dataset):
#     def __init__(self, split, tokenizer, max_length=128):
#         """
#         Initialize the IMDB dataset.

#         Args:
#         split (str): Dataset split, either "train" or "test".
#         tokenizer (BertTokenizer): Tokenizer for text encoding.
#         max_length (int): Maximum length for text sequences (default: 128).
#         """
#         self.imdb = list(IMDB(split=split))
#         self.tokenizer = tokenizer
#         self.max_length = max_length
        
#         # Extract texts and labels
#         self.data = [text for label, text in self.imdb]
#         self.targets = np.array([0 if label == 'neg' else 1 for label, _ in self.imdb])  # Convert labels to 0 (neg) or 1 (pos)
#         self.classes = ['Negative', 'Positive']  # Corresponding class names

#     def __getitem__(self, index):
#         """
#         Retrieve a sample by index.

#         Args:
#         index (int): Index of the sample.

#         Returns:
#         dict: A dictionary containing input_ids, attention_mask, labels, and index.
#         """
#         text = str(self.data[index])
#         label = self.targets[index]
        
#         # Tokenize and encode the text
#         encoding = self.tokenizer.encode_plus(
#             text,
#             add_special_tokens=True,
#             max_length=self.max_length,
#             return_token_type_ids=False,
#             padding='max_length',
#             truncation=True,
#             return_attention_mask=True,
#             return_tensors='pt',
#         )
        
#         # return data = (input_ids and attention_mask), target (labels), index
#         # return (encoding['input_ids'].flatten(), encoding['attention_mask'].flatten()),torch.tensor(label, dtype=torch.long),index 

#         return {
#             'input_ids': encoding['input_ids'].flatten(),
#             'attention_mask': encoding['attention_mask'].flatten(),
#             'labels': torch.tensor(label, dtype=torch.long),
#             'index': index  # Return the sample index
#         }
    
#     def __len__(self):
#         """
#         Return the total number of samples.

#         Returns:
#         int: Number of samples in the dataset.
#         """
#         return len(self.data)

from torch.utils.data import Dataset
import torch

class MyIMDBDataset(Dataset):
    def __init__(self, hf_dataset, tokenizer, max_length=128):
        """
        初始化 IMDB 数据集。

        参数：
        hf_dataset (Dataset): Hugging Face 加载的数据集对象。
        tokenizer (BertTokenizer): 用于文本编码的分词器。
        max_length (int): 文本序列的最大长度（默认值：128）。
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = hf_dataset['text']
        self.targets = torch.tensor(hf_dataset['label'], dtype=torch.long)
        self.classes = ['Negative', 'Positive']

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

