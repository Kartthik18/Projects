# train.py
import os, json, torch
from collections import Counter
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm

from config import CLASSES, id2label, label2id
from data import PromptProcessor
from utils import seed_everything, safe_load_json, split_texts_labels, create_data_loader

MODEL_CHECKPOINT = "chandar-lab/NeoBERT"
MODEL_PATH = "./pokemon_nlp_model"

def train_model_improved(model, train_loader, val_loader, num_epochs=2, learning_rate=2e-5, device='cuda'):
    model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    scheduler = StepLR(optimizer, step_size=1, gamma=0.95)
    best_val_acc, best_state, patience, patience_ctr = 0.0, None, 1, 0

    for epoch in range(num_epochs):
        # --- Train ---
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        for input_ids, attention_mask, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)
            optimizer.zero_grad()
            out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss, logits = out.loss, out.logits
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += float(loss.item())
            correct += (logits.argmax(-1) == labels).sum().item()
            total += labels.size(0)

        train_loss = total_loss / max(1, len(train_loader))
        train_acc = correct / max(1, total)

        # --- Validate ---
        model.eval()
        vloss = 0.0
        vcorrect = 0
        vtotal = 0
        with torch.no_grad():
            for input_ids, attention_mask, labels in val_loader:
                input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)
                out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                vloss += float(out.loss.item())
                vcorrect += (out.logits.argmax(-1) == labels).sum().item()
                vtotal += labels.size(0)
        val_loss = vloss / max(1, len(val_loader))
        val_acc = vcorrect / max(1, vtotal)

        print(f"Epoch {epoch+1}: Train Loss={train_loss:.4f} Acc={train_acc:.4f} | Val Loss={val_loss:.4f} Acc={val_acc:.4f}")

        # Early stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = model.state_dict()
            patience_ctr = 0
            print(f"  ✅ New best val acc: {best_val_acc:.4f}")
        else:
            patience_ctr += 1
            print(f"  ⏳ No improvement ({patience_ctr}/{patience})")
            if patience_ctr >= patience:
                print("  🛑 Early stopping")
                break
        scheduler.step()

    if best_state is not None:
        model.load_state_dict(best_state)
    print(f"🏆 Best Val Acc: {best_val_acc:.4f}")
    return model

def main():
    seed_everything(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🔧 Using device: {device}")

    # If already trained, skip to inference use-case (kept here for convenience)
    if os.path.exists(MODEL_PATH) and os.path.exists(os.path.join(MODEL_PATH, "config.json")):
        print("✅ Found existing model — training skipped. (Run infer.py to use it.)")
        return

    # ---- Data prep ----
    processor = PromptProcessor()

    try:
        train_data = processor.load_train_prompts("train_prompts.json")
        print(f"✅ Loaded {len(train_data)} labeled prompts")
    except Exception:
        print("⚠️ train_prompts.json not found — proceeding with synthetic only")
        train_data = []

    synth = processor.generate_synthetic_prompts(num_prompts=10000)
    all_data = train_data + synth
    texts = [d["prompt"] for d in all_data]
    labels = [label2id[d["target"]] for d in all_data]

    # Stats
    counts = Counter(labels)
    print("📊 Class distribution:")
    for cid, cnt in counts.items():
        print(f"  {id2label[cid]}: {cnt}")

    tr_texts, val_texts, tr_labels, val_labels = split_texts_labels(texts, labels, test_size=0.2, seed=42)

    # ---- Tokenizer / Model ----
    tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT, trust_remote_code=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_CHECKPOINT,
        num_labels=len(CLASSES),
        id2label=id2label,
        label2id=label2id,
        trust_remote_code=True
    )

    tr_enc = tokenizer(tr_texts, truncation=True, padding=True, max_length=1024)
    val_enc = tokenizer(val_texts, truncation=True, padding=True, max_length=1024)

    train_loader = create_data_loader(tr_enc, tr_labels, batch_size=8, shuffle=True)
    val_loader = create_data_loader(val_enc, val_labels, batch_size=16, shuffle=False)

    # ---- Train ----
    model = train_model_improved(model, train_loader, val_loader, num_epochs=2, learning_rate=2e-5, device=device)

    # ---- Save ----
    os.makedirs(MODEL_PATH, exist_ok=True)
    model.save_pretrained(MODEL_PATH)
    tokenizer.save_pretrained(MODEL_PATH)
    print(f"✅ Saved model + tokenizer → {MODEL_PATH}")

if __name__ == "__main__":
    main()
