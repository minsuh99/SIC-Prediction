import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import random
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from tqdm.auto import tqdm
from netCDF4 import Dataset

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

# Load Data
climate_file = np.load("./preprocessed/feature_real_final.npz")
climate_data = climate_file['data']

sic_file = np.load("./preprocessed/NSIDC_seaice_con_199501_202412.npz")
sic_data = sic_file['sic']

# Dataset
class SeaIceDataset(Dataset):
    def __init__(self, climate_array, sic_array, window_length, start_idx, end_idx):
        self.climate = climate_array
        self.sic = sic_array
        self.L = window_length
        self.start = start_idx + self.L
        self.end = end_idx

    def __len__(self):
        return self.end - self.start + 1

    def __getitem__(self, idx):
        t = self.start + idx
        seq_climate = self.climate[t-self.L : t]   # (L, 10, 428, 300)
        target_sic = self.sic[t]  # (428, 300)
        target_sic = np.expand_dims(target_sic, axis=0)  # (1, 428, 300)
        return torch.from_numpy(seq_climate).float(), torch.from_numpy(target_sic).float()

train_dataset = SeaIceDataset(climate_data, sic_data, 12, 0, 287)
val_dataset = SeaIceDataset(climate_data, sic_data, 12, 288, 323)
test_dataset = SeaIceDataset(climate_data, sic_data, 12, 324, 359)

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, pin_memory=True)

# GRU-based Model
class SeaIceGRU(nn.Module):
    def __init__(self, input_channels=10, hidden_size=64, height=428, width=300):
        super(SeaIceGRU, self).__init__()
        self.height = height
        self.width = width
        self.hidden_size = hidden_size
        self.input_size = input_channels * height * width

        self.gru = nn.GRU(input_size=self.input_size, hidden_size=self.hidden_size, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(self.hidden_size, 1024),
            nn.ReLU(),
            nn.Linear(1024, height * width)
        )

    def forward(self, x):
        B, L, C, H, W = x.shape
        x = x.view(B, L, -1)
        gru_out, _ = self.gru(x)
        last_output = gru_out[:, -1, :]
        out = self.fc(last_output)
        return out.view(B, 1, H, W)

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model, Loss, Optimizer
model = SeaIceGRU().to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)

best_val_loss = float('inf')
num_epochs = 50

# Train & Validate
for epoch in tqdm(range(1, num_epochs + 1), desc="Training Progress"):
    model.train()
    total_train_loss = 0.0
    for seq_climate, target_sic in tqdm(train_loader, desc="Training"):
        seq_climate, target_sic = seq_climate.to(device), target_sic.to(device)
        optimizer.zero_grad()
        pred_sic = model(seq_climate)
        loss = criterion(pred_sic, target_sic)
        loss.backward()
        optimizer.step()
        total_train_loss += loss.item() * seq_climate.size(0)

    avg_train_loss = total_train_loss / len(train_loader.dataset)

    model.eval()
    total_val_loss = 0.0
    with torch.no_grad():
        for seq_climate, target_sic in tqdm(val_loader, desc="Validation"):
            seq_climate, target_sic = seq_climate.to(device), target_sic.to(device)
            pred_sic = model(seq_climate)
            loss = criterion(pred_sic, target_sic)
            total_val_loss += loss.item() * seq_climate.size(0)

    avg_val_loss = total_val_loss / len(val_loader.dataset)
    scheduler.step(avg_val_loss)

    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': avg_val_loss
        }, 'best_seaice_gru.pth')

    print(f"[Epoch {epoch}/{num_epochs}] Train Loss = {avg_train_loss:.6f} | Val Loss = {avg_val_loss:.6f} | LR = {optimizer.param_groups[0]['lr']:.2e}")

# Test
checkpoint = torch.load('best_seaice_gru.pth', map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

test_preds, test_trues = [], []
test_losses = 0.0

with torch.no_grad():
    for seq_climate, target_sic in test_loader:
        seq_climate, target_sic = seq_climate.to(device), target_sic.to(device)
        pred_sic = model(seq_climate)
        loss = criterion(pred_sic, target_sic)
        test_losses += loss.item() * seq_climate.size(0)
        test_preds.append(pred_sic.cpu().numpy())
        test_trues.append(target_sic.cpu().numpy())

avg_test_loss = test_losses / len(test_loader.dataset)
print(f"Average Test Loss = {avg_test_loss:.6f}")

# Visualization

def plot_sic_comparison(pred, true, idx, save_path=None):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    im0 = axes[0].imshow(true[0], vmin=0, vmax=1, cmap='Blues')
    axes[0].set_title(f"Ground Truth (Idx {idx})")
    axes[0].axis('off')
    plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    im1 = axes[1].imshow(pred[0], vmin=0, vmax=1, cmap='Blues')
    axes[1].set_title(f"Prediction (Idx {idx})")
    axes[1].axis('off')
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=150)
    else:
        plt.show()
    plt.close()

# 예시 시각화
sample_idx = 0
plot_sic_comparison(pred=test_preds[sample_idx], true=test_trues[sample_idx], idx=sample_idx)
