# Pokewar Hackathon — CV Module 🎮

This module contains the **Computer Vision pipeline** for Pokémon detection using **RF-DETR**.

## Structure
- `train.py` → Training script (loads dataset via Roboflow, trains RF-DETR on Pokémon images)
- `inference.py` → Inference & evaluation script (runs predictions, saves CSV results, generates annotated visualizations)
- `utils.py` → Helper functions (GPU cleanup, visualization utilities, etc.)

## Setup
```bash
pip install rfdetr==1.2.1 supervision==0.26.1 roboflow inference-gpu==0.51.7
