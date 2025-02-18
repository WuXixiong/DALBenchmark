from torch.utils.data import Dataset
from torchtext.datasets import IMDB
import numpy as np
import torch

class MyIMDBDataset(Dataset):
    def __init__(self, split, tokenizer, max_length=128):
        """
        Initialize the IMDB dataset.

        Args:
        split (str): Dataset split, either "train" or "test".
        tokenizer (BertTokenizer): Tokenizer for text encoding.
        max_length (int): Maximum length for text sequences (default: 128).
        """
        self.imdb = list(IMDB(split=split))
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Extract texts and labels
        self.data = [text for label, text in self.imdb]
        self.targets = np.array([0 if label == 'neg' else 1 for label, _ in self.imdb])  # Convert labels to 0 (neg) or 1 (pos)
        self.classes = ['Negative', 'Positive']  # Corresponding class names

    def __getitem__(self, index):
        """
        Retrieve a sample by index.

        Args:
        index (int): Index of the sample.

        Returns:
        dict: A dictionary containing input_ids, attention_mask, labels, and index.
        """
        text = str(self.data[index])
        label = self.targets[index]
        
        # Tokenize and encode the text
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
        
        # return data = (input_ids and attention_mask), target (labels), index
        # return (encoding['input_ids'].flatten(), encoding['attention_mask'].flatten()),torch.tensor(label, dtype=torch.long),index 

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long),
            'index': index  # Return the sample index
        }
    
    def __len__(self):
        """
        Return the total number of samples.

        Returns:
        int: Number of samples in the dataset.
        """
        return len(self.data)
