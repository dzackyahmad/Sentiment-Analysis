import pandas as pd
import torch
import numpy as np
import os
import ast
from torch.utils.data import Dataset
from torch import nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import (
    BertTokenizer,
    BertModel,
    Trainer,
    TrainingArguments
)
# =======================
# Load Data
# =======================
df_cleaned = pd.read_csv('data/cleaned.csv')
df_labels = pd.read_csv('data/with_labels.csv')

# Ambil teks & label
df = pd.DataFrame()
df['review'] = df_cleaned['cleaned']
df['label_text'] = df_labels['sentiment_label']

# Konversi label ke angka
label_map = {'Negative': 0, 'Neutral': 1, 'Positive': 2}
df['label'] = df['label_text'].map(label_map)

# =======================
# Tokenizer
# =======================
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# =======================
# Dataset Class
# =======================
class SentimentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_token_type_ids=False,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# =======================
# Split Data
# =======================
X_train, X_val, y_train, y_val = train_test_split(
    df['review'],
    df['label'],
    test_size=0.2,
    stratify=df['label'],
    random_state=42
)

train_dataset = SentimentDataset(X_train.tolist(), y_train.tolist(), tokenizer)
val_dataset = SentimentDataset(X_val.tolist(), y_val.tolist(), tokenizer)

# =======================
# Custom Model
# =======================
class CustomBERTModel(nn.Module):
    def __init__(self, num_labels):
        super(CustomBERTModel, self).__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.pooler_output
        drop = self.dropout(pooled)
        logits = self.classifier(drop)

        loss = None
        if labels is not None:
            loss = self.loss_fn(logits, labels)

        return {'loss': loss, 'logits': logits}

model = CustomBERTModel(num_labels=3)

# =======================
# Metric Evaluation
# =======================
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)

    accuracy = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

# =======================
# Training Setup
# =======================
training_args = TrainingArguments(
    output_dir='./results',
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    evaluation_strategy='epoch',
    save_strategy='epoch',
    learning_rate=2e-5,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics
)

# =======================
# Run Training
# =======================
print("🚀 Training started...")
trainer.train()
print("✅ Training completed.")

# Save model & tokenizer
model_dir = './saved_model'
os.makedirs(model_dir, exist_ok=True)
torch.save(model.state_dict(), f"{model_dir}/pytorch_model.bin")
tokenizer.save_pretrained(model_dir)
print(f"📦 Model saved to: {model_dir}")
