# model_utils.py
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, TensorDataset

def create_data_loader(encodings, labels, batch_size=4, shuffle=True):
    dataset = TensorDataset(
        torch.tensor(encodings['input_ids']),
        torch.tensor(encodings['attention_mask']),
        torch.tensor(labels, dtype=torch.long)
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

def train_model(model, train_loader, val_loader, num_epochs=2, lr=2e-5, device='cuda'):
    model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = StepLR(optimizer, step_size=1, gamma=0.95)
    best_val_acc, best_state = 0, None

    for epoch in range(num_epochs):
        model.train(); correct=0; total=0
        for input_ids, mask, labels in train_loader:
            input_ids, mask, labels = input_ids.to(device), mask.to(device), labels.to(device)
            optimizer.zero_grad()
            out = model(input_ids=input_ids, attention_mask=mask, labels=labels)
            out.loss.backward(); optimizer.step()
            pred = out.logits.argmax(dim=-1)
            correct += (pred==labels).sum().item(); total += labels.size(0)
        acc = correct/total
        print(f"[{epoch+1}] Train Acc={acc:.4f}")
        scheduler.step()
    if best_state: model.load_state_dict(best_state)
    return model

@torch.no_grad()
def predict_targets(model, tokenizer, prompts, device='cuda'):
    model.to(device).eval()
    preds=[]
    for p in prompts:
        inputs = tokenizer(p, return_tensors="pt", truncation=True, max_length=512).to(device)
        pred_id = model(**inputs).logits.argmax().item()
        preds.append(model.config.id2label[pred_id])
    return preds
