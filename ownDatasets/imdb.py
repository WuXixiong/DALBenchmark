from torch.utils.data import Dataset
import torch

class MyIMDBDataset(Dataset):
    def __init__(self, hf_dataset, tokenizer, max_length=128):
        """
        Initialize the IMDB dataset.

        Args:
        hf_dataset (Dataset): Hugging Face dataset object.
        tokenizer (BertTokenizer): Tokenizer used for text encoding.
        max_length (int): Maximum length of text sequences (default: 128).
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = hf_dataset['text']
        self.targets = torch.tensor(hf_dataset['label'], dtype=torch.long)
        self.classes = ['Negative', 'Positive']

    def __getitem__(self, index):
        """
        Get a sample by index.

        Args:
        index (int): Index of the sample.

        Returns:
        dict: A dictionary containing input_ids, attention_mask, labels, and index.
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
        Return the number of samples in the dataset.

        Returns:
        int: Number of samples.
        """
        return len(self.data)
