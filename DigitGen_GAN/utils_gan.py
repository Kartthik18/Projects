# utils_gan.py
import os
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import numpy as np
import torch

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def save_grid(tensor_batch, fp: str, nrow: int = 8):
    grid = vutils.make_grid(tensor_batch, padding=2, normalize=True, nrow=nrow)
    vutils.save_image(grid, fp)
    return grid  # return for optional plotting/montage

def show_progress(saved_grids, rows: int = 6, cols: int = 5, title_prefix: str = "Epoch"):
    """Quick matplotlib montage from a list of grids."""
    plt.figure(figsize=(15, 15))
    for idx, grid in enumerate(saved_grids[:rows*cols]):
        plt.subplot(rows, cols, idx + 1)
        img = np.transpose(grid.cpu().numpy(), (1, 2, 0))
        plt.imshow(img)
        plt.axis("off")
        plt.title(f"{title_prefix} {idx+1}")
    plt.tight_layout()
    plt.show()

@torch.no_grad()
def sample_fixed(netG, fixed_noise):
    netG.eval()
    imgs = netG(fixed_noise).detach().cpu()
    netG.train()
    return imgs
