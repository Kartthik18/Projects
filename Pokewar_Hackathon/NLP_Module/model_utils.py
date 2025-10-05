# utils.py
import json, os, random
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

def safe_load_json(path: str):
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except UnicodeDecodeError:
        with open(path, 'r', encoding='latin-1') as f:
            return json.load(f)

def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def split_texts_labels(texts, labels, test_size=0.2, seed=42):
    return train_test_split(texts, labels, test_size=test_size, random_state=seed, stratify=labels)

def create_data_loader(encodings, labels, batch_size=8, shuffle=True):
    input_ids = torch.tensor(encodings['input_ids'])
    attention_mask = torch.tensor(encodings['attention_mask'])
    labels = torch.tensor(labels, dtype=torch.long)
    ds = TensorDataset(input_ids, attention_mask, labels)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)

def predict_targets(model, tokenizer, prompts, device='cuda'):
    model.to(device)
    model.eval()
    preds = []
    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(device)
        with torch.no_grad():
            logits = model(**inputs).logits
            cid = int(logits.argmax().item())
            preds.append(model.config.id2label[cid])
    return preds
