from torch.utils.data import Dataset
import torch
import numpy as np
import json
from sklearn.preprocessing import MultiLabelBinarizer
from collections import defaultdict
import os

class MyRCV1Dataset(Dataset):
    def __init__(self, data_files, tokenizer, max_length=512, imbalance_factor=1.0, use_keywords=False):
        """
        Initialize the RCV1 dataset.
        
        Args:
            data_files (str or list): Path to JSON file(s) containing the dataset.
            tokenizer: Tokenizer used for text encoding.
            max_length (int): Maximum length of text sequences (default: 512).
            imbalance_factor (float): Imbalance factor to control the degree of long-tail distribution (default: 1.0).
            use_keywords (bool): Whether to include keywords in the text (default: False).
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.use_keywords = use_keywords
        
        # Load data from JSON files
        self.raw_data = []
        if isinstance(data_files, str):
            data_files = [data_files]
            
        for file_path in data_files:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    self.raw_data.append(json.loads(line.strip()))
        
        # Extract texts and labels
        self.texts = []
        self.labels = []
        
        for item in self.raw_data:
            # Combine tokens to form text
            text = ' '.join(item['doc_token'])
            
            # Optionally include keywords
            if use_keywords and item['doc_keyword']:
                keywords = ' '.join(item['doc_keyword'])
                text = text + ' [SEP] ' + keywords
                
            self.texts.append(text)
            self.labels.append(item['doc_label'])
        
        # Create label encoder for multi-label classification
        self.mlb = MultiLabelBinarizer()
        self.encoded_labels = self.mlb.fit_transform(self.labels)
        self.classes = self.mlb.classes_.tolist()
        self.num_classes = len(self.classes)
        
        print(f"Dataset loaded: {len(self.texts)} samples")
        print(f"Number of unique labels: {self.num_classes}")
        print(f"Labels: {self.classes}")
        
        # Apply imbalance if specified
        if imbalance_factor:
            self._create_imbalanced_dataset(imbalance_factor)
        else:
            self.data = self.texts
            self.targets = torch.tensor(self.encoded_labels, dtype=torch.float)
    
    def _create_imbalanced_dataset(self, imbalance_factor):
        """
        Create an imbalanced version of the dataset following a long-tail distribution.
        For multi-label datasets, this is more complex as samples can belong to multiple classes.
        """
        # Count label frequencies
        label_counts = defaultdict(int)
        for labels in self.labels:
            for label in labels:
                label_counts[label] += 1
        
        # Sort labels by frequency (most frequent first)
        sorted_labels = sorted(label_counts.items(), key=lambda x: x[1], reverse=True)
        
        # Create sampling weights based on label frequencies and imbalance factor
        label_weights = {}
        min_freq = min(label_counts.values())
        
        for i, (label, freq) in enumerate(sorted_labels):
            # Apply exponential decay for long-tail distribution
            weight = min_freq * (imbalance_factor ** (i / (len(sorted_labels) - 1)))
            label_weights[label] = min(weight / freq, 1.0)  # Ensure weight <= 1.0
        
        # Sample data based on weights
        selected_indices = []
        for i, labels in enumerate(self.labels):
            # For multi-label samples, use the maximum weight among all labels
            max_weight = max([label_weights.get(label, 1.0) for label in labels])
            if np.random.random() < max_weight:
                selected_indices.append(i)
        
        # Update dataset
        self.data = [self.texts[i] for i in selected_indices]
        self.targets = torch.tensor([self.encoded_labels[i] for i in selected_indices], dtype=torch.float)
        
        print(f"Imbalanced dataset created: {len(self.data)} samples (reduction: {1 - len(self.data)/len(self.texts):.2%})")
    
    def get_label_distribution(self):
        """
        Get the distribution of labels in the dataset.
        """
        label_dist = defaultdict(int)
        for target in self.targets:
            active_labels = torch.nonzero(target, as_tuple=True)[0]
            for label_idx in active_labels:
                label_name = self.classes[label_idx.item()]
                label_dist[label_name] += 1
        
        return dict(label_dist)
    
    def __getitem__(self, index):
        """
        Get a sample by index.
        
        Args:
            index (int): Index of the sample.
            
        Returns:
            dict: A dictionary containing input_ids, attention_mask, labels, and index.
        """
        text = self.data[index]
        labels = self.targets[index]
        
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
            'labels': labels,  # Multi-hot encoded labels
            'index': index
        }
    
    def __len__(self):
        """
        Return the number of samples in the dataset.
        """
        return len(self.data)