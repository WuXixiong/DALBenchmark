import torch
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from sklearn.metrics import accuracy_score
from torchtext.datasets import AG_NEWS
from tqdm import tqdm

# 检查是否有可用的GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. 数据准备
class AGNewsDataset(Dataset):
    def __init__(self, split, tokenizer, max_length=128):
        self.data = []
        self.labels = []
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # 加载AG_NEWS数据集
        for label, text in AG_NEWS(split=split):
            self.data.append(text)
            self.labels.append(label - 1)  # 标签从1开始，调整为0-3

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        text = str(self.data[idx])
        label = int(self.labels[idx])
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
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# 2. 加载分词器和模型
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=4)
model = model.to(device)

# 3. 加载数据集
train_dataset = AGNewsDataset(split='train', tokenizer=tokenizer)
valid_dataset = AGNewsDataset(split='test', tokenizer=tokenizer)  # AG_NEWS没有单独的验证集，使用测试集代替

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=64)

# 4. 定义优化器
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

# 5. 训练函数
def train_epoch(model, data_loader, optimizer, device):
    model.train()
    total_loss = 0
    for batch in tqdm(data_loader, desc="Training"):
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        outputs = model(input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels)
        loss = outputs.loss
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
    return total_loss / len(data_loader)

# 6. 评估函数
def eval_model(model, data_loader, device):
    model.eval()
    predictions = []
    true_labels = []
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids=input_ids,
                            attention_mask=attention_mask)
            logits = outputs.logits
            preds = torch.argmax(logits, dim=1)
            predictions.extend(preds.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())
    acc = accuracy_score(true_labels, predictions)
    return acc

# 7. 训练和评估
epochs = 10

for epoch in range(epochs):
    print(f"Epoch {epoch + 1}/{epochs}")
    train_loss = train_epoch(model, train_loader, optimizer, device)
    print(f"Training loss: {train_loss}")
    val_acc = eval_model(model, valid_loader, device)
    print(f"Validation Accuracy: {val_acc}")

# 8. 保存模型
model.save_pretrained('bert-agnews-model')
tokenizer.save_pretrained('bert-agnews-model')
