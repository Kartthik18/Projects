# Neural Machine Translation (Transformer) — Eng↔Fra (toy)

Minimal Transformer encoder–decoder trained on a small Eng–Fra pairs file (`data/eng-fra.txt`), modularized into:
- `preprocessing.py` – normalization, vocab, dataset, dataloader
- `model.py` – positional encoding + Transformer encoder/decoder
- `main.py` – training loop, greedy decode, BLEU sampling

## Setup
```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -c "import nltk; nltk.download('punkt')"
