# PokéWar Hackathon — NLP Module (NeoBERT)

This module uses **NeoBERT** for Pokémon tactical text classification:
- Input: complex military-style text prompts
- Output: target Pokémon (`pikachu`, `charizard`, `bulbasaur`, `mewtwo`) + protected classes

## Structure
- `config.py` — class labels, synonyms, filler phrases
- `data.py` — `PromptProcessor` for synthetic prompt generation
- `model_utils.py` — data loaders, training loop, prediction
- `main.py` — end-to-end pipeline (train + predict)
- `requirements.txt`

## Run
```bash
pip install -r requirements.txt
python main.py
