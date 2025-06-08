import os
import random
import glob
import re
from netCDF4 import Dataset

import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from tqdm.auto import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

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
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

## Data Load & Normalization
climate_file = np.load("./preprocessed/feature_real_final.npz")
climate_data = climate_file['data']            # (T, C, H, W)

sic_file     = np.load("./preprocessed/NSIDC_seaice_con_199501_202412.npz")
sic_data     = sic_file['sic']                 # (T, H, W)
x_coords     = sic_file['x']
y_coords     = sic_file['y']
dates        = sic_file['dates']               # list length T
mask_full    = sic_file['mask']                # (T, H, W)

# Z‐score normalize climate using training portion
climate_train = climate_data[:288]
cm_mean = climate_train.mean(axis=(0,2,3), keepdims=True)
cm_std  = climate_train.std (axis=(0,2,3), keepdims=True)
climate_data = (climate_data - cm_mean) / (cm_std + 1e-6)

## Custom Sea Ice Dataset
class SeaIceDataset(Dataset):
    def __init__(self, climate_array, sic_array, mask_array,
                 window_length, prediction_length, start_idx, end_idx):
        self.climate = climate_array
        self.sic     = sic_array
        self.mask    = mask_array
        self.L       = window_length
        self.pred_L  = prediction_length
        self.start   = start_idx + self.L
        self.end     = end_idx - (self.pred_L - 1)

    def __len__(self):
        return self.end - self.start + 1

    def __getitem__(self, idx):
        t = self.start + idx
        seq_climate = self.climate[t-self.L : t]         # (L, C, H, W)
        target_sic  = self.sic    [t : t+self.pred_L]     # (pred_L, H, W)
        mask_seq     = self.mask   [t : t+self.pred_L]     # (pred_L, H, W)

        return (
            torch.from_numpy(seq_climate).float(),
            torch.from_numpy(target_sic ).float(),
            torch.from_numpy(mask_seq    ).float()
        )

L, P = 12, 3
train_dataset = SeaIceDataset(climate_data, sic_data, mask_full, L, P, 0,   287)
val_dataset   = SeaIceDataset(climate_data, sic_data, mask_full, L, P, 288, 323)
test_dataset  = SeaIceDataset(climate_data, sic_data, mask_full, L, P, 324, 359)

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True,  pin_memory=True)
val_loader   = DataLoader(val_dataset,   batch_size=4, shuffle=False, pin_memory=True)
test_loader  = DataLoader(test_dataset,  batch_size=4, shuffle=False, pin_memory=True)

## TCN + U-Net Model Definition
class TCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1):
        super().__init__()
        pad_t = ((kernel_size - 1) // 2) * dilation
        self.conv = nn.Conv3d(
            in_channels, out_channels,
            kernel_size=(kernel_size, 1, 1),
            padding=(pad_t, 0, 0),
            dilation=(dilation, 1, 1),
            bias=False
        )
        self.bn   = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return self.relu(x)

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels,  out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)

class UNet2D(nn.Module):
    def __init__(self, in_channels, out_channels, features=[64,128,256,512]):
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
            self.ups.append(nn.ConvTranspose2d(f*2, f, 2, 2))
            self.ups.append(DoubleConv(f*2, f))

        self.final = nn.Conv2d(features[0], out_channels, 1)

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
            skip = skips[i//2]
            if x.shape != skip.shape:
                x = F.interpolate(x, size=skip.shape[2:])
            x = self.ups[i+1](torch.cat([skip, x], dim=1))
        return self.final(x)

class SeaIceUNetTCN(nn.Module):
    def __init__(self, input_channels=10, tcn_channels=64,
                 tcn_layers=3, unet_features=[64,128,256,512],
                 pred_length=3):
        super().__init__()
        layers = []
        ch = input_channels
        for i in range(tcn_layers):
            layers.append(
                TCNBlock(ch, tcn_channels, kernel_size=3, dilation=2**i)
            )
            ch = tcn_channels
        self.tcn = nn.Sequential(*layers)
        self.unet = UNet2D(tcn_channels, pred_length, unet_features)

    def forward(self, x):
        # x: (B, L, C, H, W) → (B, C, L, H, W)
        x = x.permute(0,2,1,3,4)
        x = self.tcn(x)           # (B, tcn_channels, L, H, W)
        x = x[:,:, -1, :, :]      # last time → (B, tcn_channels, H, W)
        return self.unet(x)       # (B, pred_L, H, W)

model = SeaIceUNetTCN(10, 64, 3, [64,128,256,512], P).to(device)

## Loss, Optimizer & Scheduler
criterion = nn.MSELoss(reduction='none')
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', patience=3, factor=0.5
)

## Training with Early Stopping
best_val_loss, wait = float('inf'), 0
for epoch in range(1, 51):
    model.train()
    total_train_loss = 0.0
    for seq_climate, target_sic, mask_seq in tqdm(train_loader, desc="training"):
        seq_climate = seq_climate.to(device)
        target_sic  = target_sic.to(device)
        mask_seq    = mask_seq.to(device)

        optimizer.zero_grad()
        pred_sic = model(seq_climate)                       # (B, P, H, W)
        loss_map = criterion(pred_sic, target_sic)           # (B, P, H, W)
        loss = (loss_map * mask_seq).sum() / mask_seq.sum()
        loss.backward()
        optimizer.step()

        total_train_loss += loss.item() * seq_climate.size(0)
    avg_train_loss = total_train_loss / len(train_loader.dataset)

    model.eval()
    total_val_loss = 0.0
    with torch.no_grad():
        for seq_climate, target_sic, mask_seq in tqdm(val_loader, desc="validation"):
            seq_climate = seq_climate.to(device)
            target_sic  = target_sic.to(device)
            mask_seq    = mask_seq.to(device)

            pred_sic = model(seq_climate)
            loss_map = criterion(pred_sic, target_sic)
            loss = (loss_map * mask_seq).sum() / mask_seq.sum()

            total_val_loss += loss.item() * seq_climate.size(0)
    avg_val_loss = total_val_loss / len(val_loader.dataset)

    scheduler.step(avg_val_loss)
    if avg_val_loss < best_val_loss:
        best_val_loss, wait = avg_val_loss, 0
        torch.save(model.state_dict(), 'best_seaice_unet_tcn.pth')
    else:
        wait += 1
        if wait >= 5:
            print("Early stopping at epoch", epoch)
            break

    print(f"[Epoch {epoch}] Train Loss = {avg_train_loss:.6f} | Val Loss = {avg_val_loss:.6f}")

## Test & Visualization
model.load_state_dict(torch.load('best_seaice_unet_tcn.pth'))
model.eval()

test_preds, test_trues = [], []
test_losses = 0.0
with torch.no_grad():
    for seq_climate, target_sic, mask_seq in test_loader:
        seq_climate = seq_climate.to(device)
        target_sic  = target_sic.to(device)
        mask_seq    = mask_seq.to(device)

        pred_sic = model(seq_climate)
        loss_map = criterion(pred_sic, target_sic)
        loss = (loss_map * mask_seq).sum() / mask_seq.sum()
        test_losses += loss.item() * seq_climate.size(0)

        test_preds.append(pred_sic.cpu().numpy())
        test_trues.append(target_sic.cpu().numpy())

avg_test_loss = test_losses / len(test_loader.dataset)
print(f"Average Test Loss = {avg_test_loss:.6f}")

# Visualization helper (no L or globals)
def plot_sic_error_map(pred, true, x_coords, y_coords, dates, start_idx, sample_idx=0, horizon_step=0):
    tm = true[sample_idx][horizon_step]
    pm = pred[sample_idx][horizon_step]
    dm = tm - pm
    date_str = dates[start_idx + sample_idx + horizon_step]
    m = abs(dm).max()
    vmin, vmax = -m, m
    X, Y = np.meshgrid(x_coords, y_coords)
    plt.figure(figsize=(8,10))
    im = plt.pcolormesh(X, Y, dm, cmap='bwr', vmin=vmin, vmax=vmax, shading='auto')
    plt.colorbar(im, label='True − Predicted')
    plt.title(f'SIC Error – {date_str}, Horizon {horizon_step+1}')
    plt.xlabel('X (km)'); plt.ylabel('Y (km)'); plt.tight_layout(); plt.show()

test_pred_arr = np.concatenate(test_preds, axis=0)
test_true_arr = np.concatenate(test_trues, axis=0)
plot_sic_error_map(test_pred_arr, test_true_arr, x_coords, y_coords, dates, test_dataset.start, 0, 0)
