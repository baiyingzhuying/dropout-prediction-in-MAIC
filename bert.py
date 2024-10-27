import json
import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import DataLoader, Dataset, random_split
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score

# 自定义数据集类
class CustomDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        tokens = self.tokenizer(text, add_special_tokens=True, padding='max_length', truncation=True, max_length=self.max_length, return_tensors="pt")
        input_ids = tokens['input_ids'].squeeze()
        attention_mask = tokens['attention_mask'].squeeze()
        return input_ids, attention_mask, label

# 读取JSON文件并提取数据
def load_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    texts = []
    labels = []
    for item in data:
        history_content = " ".join(item['HistoryContent'])
        texts.append(history_content)
        labels.append(int(item['GroundTruth']))
    return texts, labels

# 加载训练和测试数据
train_texts, train_labels = load_data('train.json')
test_texts, test_labels = load_data('test.json')

# 加载预训练的BERT模型和分词器
model_name = "bert-base-chinese"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)

# 创建数据集和数据加载器
train_dataset = CustomDataset(train_texts, train_labels, tokenizer, max_length=128)
test_dataset = CustomDataset(test_texts, test_labels, tokenizer, max_length=128)

# 拆分训练集和验证集
train_size = int(0.875 * len(train_dataset))
valid_size = len(train_dataset) - train_size
train_subset, valid_subset = random_split(train_dataset, [train_size, valid_size])

# 创建数据加载器
train_loader = DataLoader(train_subset, batch_size=8, shuffle=True)
valid_loader = DataLoader(valid_subset, batch_size=8, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

# 训练模型
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
best_valid_loss = float('inf')
early_stopping_patience = 3
patience_counter = 0
best_epoch = 0

model.train()
for epoch in range(20):  # 假设最多训练20个epoch
    print(f"Epoch {epoch} Training:")
    for input_ids, attention_mask, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        print(f"Loss: {loss.item()}")

    # 验证模型
    model.eval()
    valid_loss = 0
    with torch.no_grad():
        for input_ids, attention_mask, labels in valid_loader:
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            valid_loss += loss.item()
    avg_valid_loss = valid_loss / len(valid_loader)
    print(f"Epoch {epoch}, Validation Loss: {avg_valid_loss}")

    # 早停机制
    if avg_valid_loss < best_valid_loss:
        best_valid_loss = avg_valid_loss
        best_epoch = epoch
        patience_counter = 0
        # 保存最佳模型
        torch.save(model.state_dict(), 'best_model.pt')
    else:
        patience_counter += 1
        if patience_counter >= early_stopping_patience:
            print(f"Early stopping at epoch {epoch}")
            break
    model.train()

# 加载最佳模型
model.load_state_dict(torch.load('best_model.pt'))

# 测试模型
model.eval()
all_preds = []
all_labels = []
with torch.no_grad():
    for input_ids, attention_mask, labels in test_loader:
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        preds = torch.argmax(outputs.logits, dim=-1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# 计算预测准确率和其他指标
accuracy = accuracy_score(all_labels, all_preds)
precision = precision_score(all_labels, all_preds)
recall = recall_score(all_labels, all_preds)
f1 = f1_score(all_labels, all_preds)
conf_matrix = confusion_matrix(all_labels, all_preds)
tn, fp, fn, tp = conf_matrix.ravel()

print(f"Test Accuracy: {accuracy}")
print(f"Test Precision: {precision}, Recall: {recall}, F1 Score: {f1}")
print(f"Confusion Matrix:\n{conf_matrix}")
print(f"True Negatives (TN): {tn}")
print(f"False Positives (FP): {fp}")
print(f"False Negatives (FN): {fn}")
print(f"True Positives (TP): {tp}")
print(f"Training stopped at epoch {best_epoch}")