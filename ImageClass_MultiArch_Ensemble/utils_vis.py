# utils_vis.py
import matplotlib.pyplot as plt
import numpy as np
import torch

def imshow(img_tensor):
    # Unnormalize from [-1,1] back to [0,1]
    img = img_tensor / 2 + 0.5
    npimg = img.detach().cpu().numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.axis('off')
    plt.show()

@torch.no_grad()
def preview_batch(models, images, labels, classes, device):
    for m in models:
        m.eval()
    preds = None
    # Use first model to compute embeddings? Not needed—just ensemble
    from train import ensemble_predict
    preds = ensemble_predict(models, images.to(device))
    for i in range(min(5, images.size(0))):
        imshow(images[i])
        print(f"Actual: {classes[labels[i]]} | Ensemble Predicted: {classes[preds[i]]}")
