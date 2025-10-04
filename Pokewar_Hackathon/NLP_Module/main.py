# main.py
import os, json, random
import torch
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from collections import Counter
from config import CLASSES, label2id, id2label
from data import PromptProcessor
from model_utils import create_data_loader, train_model, predict_targets

MODEL_PATH = "./pokemon_nlp_model"
MODEL_CHECKPOINT = "chandar-lab/NeoBERT"

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("🚀 PokéWar NLP (NeoBERT) on", device)

    processor = PromptProcessor()
    try:
        train_data = processor.load_train_prompts("train_prompts.json")
    except:
        train_data = []
    synthetic_data = processor.generate_synthetic_prompts(2000)
    all_data = train_data + synthetic_data
    random.shuffle(all_data)

    texts = [d['prompt'] for d in all_data]
    labels = [label2id[d['target']] for d in all_data]
    train_texts, val_texts, y_train, y_val = train_test_split(texts, labels, test_size=0.2, stratify=labels)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT, trust_remote_code=True)
    train_enc = tokenizer(train_texts, truncation=True, padding=True)
    val_enc = tokenizer(val_texts, truncation=True, padding=True)

    train_loader = create_data_loader(train_enc, y_train, batch_size=4)
    val_loader   = create_data_loader(val_enc, y_val, batch_size=8, shuffle=False)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_CHECKPOINT, num_labels=len(CLASSES),
        id2label=id2label, label2id=label2id, trust_remote_code=True
    )

    model = train_model(model, train_loader, val_loader, num_epochs=2, device=device)

    os.makedirs(MODEL_PATH, exist_ok=True)
    model.save_pretrained(MODEL_PATH)
    tokenizer.save_pretrained(MODEL_PATH)

    # Test on dummy prompts
    prompts = ["Neutralize the fire-breathing lizard immediately.", "Protect the vine pokemon at all costs."]
    preds = predict_targets(model, tokenizer, prompts, device)
    for p, pr in zip(prompts, preds):
        print(f"Prompt: {p} → Predicted: {pr}")

if __name__ == "__main__":
    main()
