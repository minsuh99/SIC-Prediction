import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import random
import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from netCDF4 import Dataset

import torch
import torch.nn as nn
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
        seq_climate = self.climate[t-self.L : t]
        target_sic = self.sic[t]
        target_sic = np.expand_dims(target_sic, axis=0)
        return torch.from_numpy(seq_climate).float(), torch.from_numpy(target_sic).float()

train_dataset = SeaIceDataset(climate_data, sic_data, 12, 0, 287)
val_dataset = SeaIceDataset(climate_data, sic_data, 12, 288, 323)
test_dataset = SeaIceDataset(climate_data, sic_data, 12, 324, 359)

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, pin_memory=True)

# GRU Regression Model
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
            nn.Linear(1024, 1)
        )

    def forward(self, x):
        B, L, C, H, W = x.shape
        x = x.view(B, L, -1)
        gru_out, _ = self.gru(x)
        last_output = gru_out[:, -1, :]
        out = self.fc(last_output)
        return out.view(B, 1)

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SeaIceGRU().to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)

best_val_loss = float('inf')
num_epochs = 50

# Train & Validation
for epoch in tqdm(range(1, num_epochs + 1), desc="Training Progress"):
    model.train()
    total_train_loss = 0.0
    for seq_climate, target_sic in tqdm(train_loader, desc="Training"):
        seq_climate, target_sic = seq_climate.to(device), target_sic.to(device)
        target_scalar = target_sic.mean(dim=(1, 2, 3), keepdim=True)
        optimizer.zero_grad()
        pred_scalar = model(seq_climate)
        loss = criterion(pred_scalar, target_scalar)
        loss.backward()
        optimizer.step()
        total_train_loss += loss.item() * seq_climate.size(0)

    avg_train_loss = total_train_loss / len(train_loader.dataset)

    model.eval()
    total_val_loss = 0.0
    with torch.no_grad():
        for seq_climate, target_sic in tqdm(val_loader, desc="Validation"):
            seq_climate, target_sic = seq_climate.to(device), target_sic.to(device)
            target_scalar = target_sic.mean(dim=(1, 2, 3), keepdim=True)
            pred_scalar = model(seq_climate)
            loss = criterion(pred_scalar, target_scalar)
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
        }, 'best_seaice_gru_regression.pth')

    print(f"[Epoch {epoch}/{num_epochs}] Train Loss = {avg_train_loss:.6f} | Val Loss = {avg_val_loss:.6f} | LR = {optimizer.param_groups[0]['lr']:.2e}")

# Test
checkpoint = torch.load('best_seaice_gru_regression.pth', map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

test_preds, test_trues = [], []
test_losses = 0.0

with torch.no_grad():
    for seq_climate, target_sic in test_loader:
        seq_climate, target_sic = seq_climate.to(device), target_sic.to(device)
        target_scalar = target_sic.mean(dim=(1, 2, 3), keepdim=True)
        pred_scalar = model(seq_climate)
        loss = criterion(pred_scalar, target_scalar)
        test_losses += loss.item() * seq_climate.size(0)
        test_preds.append(pred_scalar.cpu().numpy())
        test_trues.append(target_scalar.cpu().numpy())

avg_test_loss = test_losses / len(test_loader.dataset)
print(f"Average Test Loss = {avg_test_loss:.6f}")

# Visualization (for regression)
def plot_scalar_comparison(pred_scalar, true_scalar, idx, save_path=None):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(['Ground Truth', 'Prediction'], [true_scalar[0][0], pred_scalar[0][0]], color=['blue', 'orange'])
    ax.set_ylim(0, 1)
    ax.set_ylabel('SIC Mean')
    ax.set_title(f'Scalar SIC Prediction (Idx {idx})')
    for i, v in enumerate([true_scalar[0][0], pred_scalar[0][0]]):
        ax.text(i, v + 0.01, f"{v:.4f}", ha='center')
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=150)
    else:
        plt.show()
    plt.close()

# 예시 시각화
sample_idx = 0
plot_scalar_comparison(pred_scalar=test_preds[sample_idx], true_scalar=test_trues[sample_idx], idx=sample_idx)
