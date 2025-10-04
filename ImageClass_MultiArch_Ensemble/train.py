# train.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

def train_model(model, trainloader, optimizer, criterion, device, num_epochs=10):
    for epoch in range(num_epochs):
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        for inputs, labels in tqdm(trainloader, desc=f"Epoch {epoch+1}", leave=False):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

        print(f"[{epoch+1:02d}] Train Loss: {running_loss/len(trainloader):.4f} | Acc: {100*correct/total:.2f}%")

def ensemble_predict(models, input_tensor):
    # Average softmax probabilities from each model
    probs = [F.softmax(m(input_tensor), dim=1) for m in models]
    avg = torch.stack(probs).mean(dim=0)
    _, pred = torch.max(avg, 1)
    return pred

@torch.no_grad()
def test_ensemble(models, loader, device):
    for m in models:
        m.eval()
    correct, total = 0, 0
    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)
        preds = ensemble_predict(models, inputs)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    print(f"Ensemble Test Accuracy: {100.0 * correct / total:.2f}%")
