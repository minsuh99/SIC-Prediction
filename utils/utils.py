import os
import random
import numpy as np
from matplotlib.colors import BoundaryNorm
import matplotlib.pyplot as plt
import torch
from tqdm.auto import tqdm


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
# Visualization
def visualize_pred_L_1(pred, true, mask, x_coords, y_coords, dates, start_idx, save_dir):
    os.makedirs(save_dir, exist_ok=True)

    N, L, _, _ = pred.shape
    X, Y = np.meshgrid(x_coords, y_coords)

    for i in tqdm(range(N), desc="Visualizing...", leave=False):
        fig, ax = plt.subplots(1, 1, figsize=(8, 10))
        diff_map = true[i, 0] - pred[i, 0]
        masked_diff_map = np.where(mask[i, 0] == 1, diff_map, np.nan)
        date_str = dates[start_idx + i]

        cmap = plt.get_cmap('bwr').copy()
        cmap.set_bad(color='gray')

        bounds = np.linspace(-1, 1, 17)
        norm = BoundaryNorm(boundaries=bounds, ncolors=256)

        im = ax.pcolormesh(X, Y, masked_diff_map, cmap=cmap, norm=norm)
        ax.set_title(f'{date_str}', fontsize=14)
        ax.set_xlabel('X (km)')
        ax.set_ylabel('Y (km)')

        fig.subplots_adjust(right=0.92)
        cbar_ax = fig.add_axes([0.94, 0.15, 0.02, 0.7])
        plt.colorbar(im, cax=cbar_ax, label='True − Predicted')

        input_start = dates[start_idx + i - 12]
        input_end = dates[start_idx + i - 1]
        fig.suptitle(f'Sample {i+1}: Input ({input_start} ~ {input_end}) → 1-Month Prediction', fontsize=16)

        pred_date = dates[start_idx + i]
        fname = f'sic_error_map_{pred_date}_sample_{i+1:03d}.png'
        plt.savefig(os.path.join(save_dir, fname), bbox_inches='tight')
        plt.close()

# Visualization (pred_L=3)
def visualize_pred_L_3(pred, true, mask, x_coords, y_coords, dates, start_idx, save_dir):
    os.makedirs(save_dir, exist_ok=True)

    N, L, _, _ = pred.shape
    X, Y = np.meshgrid(x_coords, y_coords)

    for i in tqdm(range(N), desc="Visualizing...", leave=False):
        fig, axes = plt.subplots(1, 3, figsize=(24, 10))
        
        for h in range(L):
            diff_map = true[i, h] - pred[i, h]
            masked_diff_map = np.where(mask[i, h] == 1, diff_map, np.nan)
            date_str = dates[start_idx + i + h]

            cmap = plt.get_cmap('bwr').copy()
            cmap.set_bad(color='gray')

            bounds = np.linspace(-1, 1, 17)
            norm = BoundaryNorm(boundaries=bounds, ncolors=256)

            im = axes[h].pcolormesh(X, Y, masked_diff_map, cmap=cmap, norm=norm)
            axes[h].set_title(f'{date_str}', fontsize=14)
            axes[h].set_xlabel('X (km)')
            axes[h].set_ylabel('Y (km)')

        fig.subplots_adjust(right=0.92)
        cbar_ax = fig.add_axes([0.94, 0.15, 0.02, 0.7])
        plt.colorbar(im, cax=cbar_ax, label='True − Predicted')

        input_start = dates[start_idx + i - 12]
        input_end = dates[start_idx + i - 1]
        fig.suptitle(f'Sample {i+1}: Input ({input_start} ~ {input_end}) → 3-Month Prediction', fontsize=16)

        pred_start = dates[start_idx + i]
        pred_end = dates[start_idx + i + 2]
        fname = f'sic_error_map_{pred_start}_to_{pred_end}_sample_{i+1:03d}.png'
        plt.savefig(os.path.join(save_dir, fname), bbox_inches='tight')
        plt.close()
        
# Visualization (pred_L=6)
def visualize_pred_L_6(pred, true, mask, x_coords, y_coords, dates, start_idx, save_dir):
    os.makedirs(save_dir, exist_ok=True)

    N, L, _, _ = pred.shape
    X, Y = np.meshgrid(x_coords, y_coords)

    for i in tqdm(range(N), desc="Visualizing...", leave=False):
        fig, axes = plt.subplots(1, L, figsize=(6 * L, 10))

        for h in range(L):
            diff_map = true[i, h] - pred[i, h]
            masked_diff_map = np.where(mask[i, h] == 1, diff_map, np.nan)
            date_str = dates[start_idx + i + h]

            cmap = plt.get_cmap('bwr').copy()
            cmap.set_bad(color='gray')

            bounds = np.linspace(-1, 1, 17)
            norm = BoundaryNorm(boundaries=bounds, ncolors=256)

            im = axes[h].pcolormesh(X, Y, masked_diff_map, cmap=cmap, norm=norm)
            axes[h].set_title(f'{date_str}', fontsize=14)
            axes[h].set_xlabel('X (km)')
            axes[h].set_ylabel('Y (km)')

        fig.subplots_adjust(right=0.92)
        cbar_ax = fig.add_axes([0.94, 0.15, 0.02, 0.7])
        plt.colorbar(im, cax=cbar_ax, label='True − Predicted')

        input_start = dates[start_idx + i - 12]
        input_end = dates[start_idx + i - 1]
        fig.suptitle(f'Sample {i+1}: Input ({input_start} ~ {input_end}) → 6-Month Prediction', fontsize=16)

        pred_start = dates[start_idx + i]
        pred_end = dates[start_idx + i + L - 1]
        fname = f'sic_error_map_{pred_start}_to_{pred_end}_sample_{i+1:03d}.png'
        plt.savefig(os.path.join(save_dir, fname), bbox_inches='tight')
        plt.close()