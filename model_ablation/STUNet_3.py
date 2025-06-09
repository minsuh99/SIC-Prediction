import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
import random
import glob
import re
from netCDF4 import Dataset

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from datetime import datetime
from tqdm.auto import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
# Fix Seed
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)
## Data Load
# Feature Data
climate_file = np.load("./preprocessed/feature_real_final.npz") # Download Feature Data
climate_data = climate_file['data']

# Target Data
sic_file = np.load("./preprocessed/NSIDC_seaice_con_199501_202412.npz")
sic_data = sic_file['sic']
x_coords = sic_file['x']
y_coords = sic_file['y']
dates = sic_file['dates']  # List of dates corresponding to the SIC data
mask = sic_file['mask']  # (360, 428, 300) (0: False(sea), 1: True(non-sea))

# Additional preprocessing (Z-score normalization, with train_data)
climate_train = climate_data[:240] 
# Feature-wise mean & std
cm_mean = climate_train.mean(axis=(0,2,3), keepdims=True)
cm_std = climate_train.std(axis=(0,2,3), keepdims=True)

climate_data = (climate_data - cm_mean) / (cm_std + 1e-6)

## Custom Sea Ice Dataset (Only SIC)
class SICOnlyDataset(Dataset):
    def __init__(self, sic_array, mask_array, window_length, prediction_length, start_idx, end_idx):
        self.sic = sic_array
        self.mask = mask_array
        self.L = window_length
        self.pred_L = prediction_length
        self.start = start_idx + self.L
        self.end = end_idx - (self.pred_L - 1)

    def __len__(self):
        return self.end - self.start + 1

    def __getitem__(self, idx):
        t = self.start + idx
        seq_sic = self.sic[t - self.L : t]              # (L, 428, 300)
        target_sic = self.sic[t : t + self.pred_L]      # (pred_L, 428, 300)
        mask = self.mask[t : t + self.pred_L]           # (pred_L, 428, 300)

        seq_sic = torch.from_numpy(seq_sic).unsqueeze(1).float()  # (L, 1, 428, 300)
        target_sic = torch.from_numpy(target_sic).float()
        mask = torch.from_numpy(mask).float()
        valid_mask = 1.0 - mask

        return seq_sic, target_sic, valid_mask

train_dataset = SICOnlyDataset(sic_array=sic_data, mask_array=mask, window_length=12, prediction_length=3, start_idx=0, end_idx=239)
val_dataset = SICOnlyDataset(sic_array=sic_data, mask_array=mask, window_length=12, prediction_length=3, start_idx=240, end_idx=299)
test_dataset = SICOnlyDataset(sic_array=sic_data, mask_array=mask, window_length=12, prediction_length=3, start_idx=300, end_idx=359)

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=2, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=2, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=2, pin_memory=True)

## TCN + U-Net Model Definition (Code by GPT)
class TCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1):
        super().__init__()
        self.dilation = dilation
        self.kernel_size = kernel_size
        self.conv = nn.Conv3d(
            in_channels, out_channels,
            kernel_size=(kernel_size, 1, 1),
            padding=0,
            dilation=(dilation, 1, 1),
            bias=False
        )
        self.bn = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout3d(0.2)
        self.residual = nn.Conv3d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        pad_amt = self.dilation * (self.kernel_size - 1)
        res = self.residual(x)

        # pad only time dimension (dim=2)
        x = F.pad(x, (0, 0, 0, 0, pad_amt, 0))  # pad only before
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.dropout(x)

        return x + res

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels,  out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.2),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.2)
        )

    def forward(self, x):
        return self.net(x)

class UNet2D(nn.Module):
    def __init__(self, in_channels, out_channels, features=[64, 128, 256, 512]):
        super().__init__()
        self.downs = nn.ModuleList()
        ch = in_channels
        for f in features:
            self.downs.append(DoubleConv(ch, f))
            ch = f
        self.pool = nn.MaxPool2d(2,2)

        self.bottleneck = DoubleConv(features[-1], features[-1]*2)

        self.ups = nn.ModuleList()
        for f in reversed(features):
            self.ups.append(nn.ConvTranspose2d(f * 2, f, 2, 2))
            self.ups.append(DoubleConv(f * 2, f))

        self.final = nn.Conv2d(features[0], out_channels, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        skips = []
        for down in self.downs:
            x = down(x)
            skips.append(x)
            x = self.pool(x)
        x = self.bottleneck(x)
        skips = skips[::-1]
        for i in range(0, len(self.ups), 2):
            x = self.ups[i](x)
            skip = skips[i // 2]
            if x.shape != skip.shape:
                x = F.interpolate(x, size=skip.shape[2:])
            x = self.ups[i + 1](torch.cat([skip, x], dim=1))
        x = self.final(x)
        x = self.sigmoid(x)
        return x

class SeaIceSTUNet(nn.Module):
    def __init__(self, input_channels=1, tcn_channels=64, tcn_layers=3, unet_features=[64, 128, 256, 512], pred_L=3):
        super().__init__()
        self.pred_L = pred_L
        
        layers = []
        ch = input_channels
        for i in range(tcn_layers):
            layers.append(TCNBlock(ch, tcn_channels, kernel_size=3, dilation=2**i))
            ch = tcn_channels
        self.tcn = nn.Sequential(*layers)
        self.unet = UNet2D(tcn_channels, pred_L, unet_features)

    def forward(self, x):
        # x: (B, L, C, H, W) → (B, C, L, H, W)
        x = x.permute(0,2,1,3,4)
        x = self.tcn(x)           # (B, tcn_channels, L, H, W)
        x = x[:,:, -1, :, :]      # last time → (B, tcn_channels, H, W)
        return self.unet(x)       # (B, pred_L, H, W)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_epochs = 30
model = SeaIceSTUNet(input_channels=1, tcn_channels=64, tcn_layers=3, unet_features=[64,128,256,512], pred_L=3).to(device)

criterion = nn.MSELoss(reduction='none')
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)

best_val_loss = float('inf')

## Train & Validation
for epoch in tqdm(range(1, num_epochs+1), desc="Training Progress", leave=True):
    # Train
    model.train()
    total_train_loss = 0.0
    for seq_climate, target_sic, mask in tqdm(train_loader, desc="training", leave=False):
        seq_climate = seq_climate.to(device) # (B, L, 10, 428, 300)
        target_sic  = target_sic.to(device) # (B, pred_L, 428, 300)
        mask = mask.to(device) # (B, pred_L, 428, 300)

        optimizer.zero_grad()
        pred_sic = model(seq_climate)            # (B, pred_L, 428, 300)
        loss_map = criterion(pred_sic, target_sic)
        loss = (loss_map * mask).sum() / mask.sum()
        loss.backward()
        optimizer.step()
        total_train_loss += loss.item() * seq_climate.size(0)

    avg_train_loss = total_train_loss / len(train_loader.dataset)

    # Validation
    model.eval()
    total_val_loss = 0.0
    with torch.no_grad():
        for seq_climate, target_sic, mask in tqdm(val_loader, desc="validation", leave=False):
            seq_climate = seq_climate.to(device)
            target_sic = target_sic.to(device)
            mask = mask.to(device) # (B, pred_L, 428, 300)

            pred_sic = model(seq_climate)
            loss_map = criterion(pred_sic, target_sic)
            loss = (loss_map * mask).sum() / mask.sum()
            total_val_loss += loss.item() * seq_climate.size(0)

    avg_val_loss = total_val_loss / len(val_loader.dataset)

    scheduler.step(avg_val_loss)
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), 'best_seaice_STUNet_3_SICOnly.pth')

    tqdm.write(f"[Epoch {epoch}/{num_epochs}] Train Loss = {avg_train_loss:.6f}  |  Val Loss = {avg_val_loss:.6f}  |  LR = {optimizer.param_groups[0]['lr']:.2e}")

## Test & Visualization
model.load_state_dict(torch.load('best_seaice_STUNet_3_SICOnly.pth', map_location=device))
model.eval()

test_preds = []
test_trues = []
test_losses = 0.0
test_mae_total = 0.0
test_mse_total = 0.0
total_valid_pixels = 0

with torch.no_grad():
    for seq_climate, target_sic, mask in test_loader:
        seq_climate = seq_climate.to(device)
        target_sic = target_sic.to(device)
        mask = mask.to(device)

        pred_sic = model(seq_climate)
        loss_map = criterion(pred_sic, target_sic)
        loss = (loss_map * mask).sum() / mask.sum()
        test_losses += loss.item() * seq_climate.size(0)

        mae_map = F.l1_loss(pred_sic, target_sic, reduction='none')
        test_mae_total += (mae_map * mask).sum().item()
        test_mse_total += (loss_map * mask).sum().item()
        total_valid_pixels += mask.sum().item()

        test_preds.append(pred_sic.cpu().numpy())
        test_trues.append(target_sic.cpu().numpy())

avg_test_loss = test_losses / len(test_loader.dataset)
avg_test_mae = test_mae_total / total_valid_pixels
avg_test_mse = test_mse_total / total_valid_pixels
avg_test_rmse = avg_test_mse ** 0.5

print(f"Average Test MSE = {avg_test_mse:.6f}")
print(f"Average Test RMSE = {avg_test_rmse:.6f}")
print(f"Average Test MAE = {avg_test_mae:.6f}")

def plot_sic_error_map(pred, true, mask, x_coords, y_coords, dates, start_idx, save_dir='./results/STUNet_3_SICOnly'):
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
            norm = matplotlib.colors.BoundaryNorm(boundaries=bounds, ncolors=256)

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


# Print Visualization
L = test_dataset.L
test_start_idx = test_dataset.start
test_pred_vis = np.concatenate(test_preds, axis=0)
test_true_vis = np.concatenate(test_trues, axis=0)
test_mask_vis = torch.cat([m for _, _, m in test_loader], dim=0).numpy()

# Visualize the prediction_step (0, first month) in Pred_L(3 months)
plot_sic_error_map(pred=test_pred_vis, true=test_true_vis, mask=test_mask_vis, x_coords=x_coords, y_coords=y_coords, dates=dates, start_idx=test_start_idx)
