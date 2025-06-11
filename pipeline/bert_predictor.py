# pipeline/bert_predictor.py

import torch
from transformers import BertTokenizer, BertModel
import torch.nn as nn
import os

# Harus sesuai dengan model arsitektur custom
class CustomBERTModel(nn.Module):
    def __init__(self, num_labels=3):
        super(CustomBERTModel, self).__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.pooler_output
        drop = self.dropout(pooled)
        logits = self.classifier(drop)
        return logits

def load_model(model_dir="saved_model"):
    # Muat tokenizer
    tokenizer = BertTokenizer.from_pretrained(model_dir)

    # Muat model dan weight
    model = CustomBERTModel()
    model.load_state_dict(torch.load(os.path.join(model_dir, "pytorch_model.bin"), map_location=torch.device("cpu")))
    model.eval()

    return model, tokenizer

def predict_sentiment(text, model, tokenizer):
    encoding = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=128,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_token_type_ids=False,
        return_tensors='pt'
    )

    input_ids = encoding['input_ids']
    attention_mask = encoding['attention_mask']

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        probs = torch.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probs, dim=1)

    label_map = {0: "Negative", 1: "Neutral", 2: "Positive"}
    return label_map[predicted.item()], confidence.item()
