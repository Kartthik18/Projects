# Image Classification — GoogLeNet, ResNet18, VGG11 + Ensemble (CIFAR-10)

Trains three architectures on CIFAR-10 and evaluates an **ensemble** by averaging softmax probabilities.

## Files
- `data.py` — loaders & transforms (96×96 resize, normalize to [-1,1])
- `models.py` — GoogLeNet/ResNet18/VGG11 heads adjusted for 10 classes
- `train.py` — single-model trainer + ensemble inference/testing
- `utils_vis.py` — quick image preview helper
- `main.py` — orchestrates training and ensemble evaluation

## Run
```bash
python -m venv .venv && source .venv/bin/activate
pip install torch torchvision tqdm matplotlib numpy
python ImageClass_MultiArch_Ensemble/main.py
