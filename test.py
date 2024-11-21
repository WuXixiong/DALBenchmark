import torch
from torchtext.datasets import IMDB
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torch.utils.data import DataLoader
from transformers import (
    AutoModelWithLMHead,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    AutoModelForMaskedLM,
    AutoConfig
)

# 设置 tokenizer 和 Vocabulary
tokenizer = get_tokenizer("basic_english")

# 定义创建词汇表的函数
def yield_tokens(data_iter):
    for _, text in data_iter:
        yield tokenizer(text)

# 加载训练和测试数据集
train_iter = IMDB(split='train')
vocab = build_vocab_from_iterator(yield_tokens(train_iter), specials=["<unk>"])
vocab.set_default_index(vocab["<unk>"])

# 定义文本转换函数
def text_pipeline(x):
    return vocab(tokenizer(x))

# 定义标签转换函数
def label_pipeline(x):
    return 1 if x == 'pos' else 0

# 定义数据加载器函数
def collate_batch(batch):
    label_list, text_list, offsets = [], [], [0]
    for label, text in batch:
        label_list.append(label_pipeline(label))
        processed_text = torch.tensor(text_pipeline(text), dtype=torch.int64)
        text_list.append(processed_text)
        offsets.append(processed_text.size(0))
    label_list = torch.tensor(label_list, dtype=torch.int64)
    offsets = torch.tensor(offsets[:-1]).cumsum(dim=0)
    text_list = torch.cat(text_list)
    return label_list, text_list, offsets

train_iter, test_iter = IMDB()
train_dataloader = DataLoader(train_iter, batch_size=8, shuffle=True, collate_fn=collate_batch)
test_dataloader = DataLoader(test_iter, batch_size=8, shuffle=True, collate_fn=collate_batch)

import torch
import torch.nn as nn

class TextClassificationModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_class):
        super(TextClassificationModel, self).__init__()
        self.embedding = nn.EmbeddingBag(vocab_size, embed_dim, sparse=True)
        self.fc = nn.Linear(embed_dim, num_class)
        self.init_weights()

    def init_weights(self):
        initrange = 0.5
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.fc.weight.data.uniform_(-initrange, initrange)
        self.fc.bias.data.zero_()

    def forward(self, text, offsets):
        embedded = self.embedding(text, offsets)
        return self.fc(embedded)

# 初始化模型
num_class = 2  # pos or neg
vocab_size = len(vocab)
embed_dim = 64
model = TextClassificationModel(vocab_size, embed_dim, num_class)

import torch.optim as optim

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001)

def train(dataloader):
    model.train()
    total_acc, total_count = 0, 0
    for idx, (label, text, offsets) in enumerate(dataloader):
        optimizer.zero_grad()
        predicted_label = model(text, offsets)
        loss = criterion(predicted_label, label)
        loss.backward()
        optimizer.step()
        total_acc += (predicted_label.argmax(1) == label).sum().item()
        total_count += label.size(0)
    print(f"Train Accuracy: {total_acc / total_count:.4f}")

def evaluate(dataloader):
    model.eval()
    total_acc, total_count = 0, 0
    with torch.no_grad():
        for idx, (label, text, offsets) in enumerate(dataloader):
            predicted_label = model(text, offsets)
            total_acc += (predicted_label.argmax(1) == label).sum().item()
            total_count += label.size(0)
    print(f"Test Accuracy: {total_acc / total_count:.4f}")

for epoch in range(10):
    print(f"Epoch {epoch + 1}")
    train(train_dataloader)

evaluate(test_dataloader)
