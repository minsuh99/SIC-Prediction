import os
import random
import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

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
        seq_climate = self.climate[t-self.L:t]  # (L, 10, H, W)
        target_sic = self.sic[t]  # (H, W)
        target_sic = np.expand_dims(target_sic, axis=0)  # (1, H, W)
        seq_climate = torch.from_numpy(seq_climate).float()
        target_sic = torch.from_numpy(target_sic).float()
        return seq_climate, target_sic

train_dataset = SeaIceDataset(climate_data, sic_data, 12, 0, 287)
val_dataset = SeaIceDataset(climate_data, sic_data, 12, 288, 323)
test_dataset = SeaIceDataset(climate_data, sic_data, 12, 324, 359)

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=0, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=0, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=0, pin_memory=True)

# GRU-based Model
class SeaIceGRU(nn.Module):
    def __init__(self, input_channels=10, hidden_dim=64, height=428, width=300):
        super(SeaIceGRU, self).__init__()
        self.height = height
        self.width = width
        self.hidden_dim = hidden_dim

        # GRU input: (B, L, C, H, W) → (B*H*W, L, C)
        self.gru = nn.GRU(input_size=input_channels, hidden_size=hidden_dim, batch_first=True)

        # Output conv head: (B, H, W, hidden_dim) → (B, 1, H, W)
        self.conv = nn.Sequential(
            nn.Conv2d(hidden_dim, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 1, kernel_size=3, padding=1)
        )

    def forward(self, x):
        B, L, C, H, W = x.shape
        x = x.permute(0, 3, 4, 1, 2).contiguous()  # (B, H, W, L, C)
        x = x.view(B * H * W, L, C)  # (B*H*W, L, C)
        out, _ = self.gru(x)  # (B*H*W, L, hidden)
        out = out[:, -1, :]  # (B*H*W, hidden)
        out = out.view(B, H, W, self.hidden_dim).permute(0, 3, 1, 2)  # (B, hidden, H, W)
        out = self.conv(out)  # (B, 1, H, W)
        return out

# Training settings
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SeaIceGRU().to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)

# Train
num_epochs = 50
best_val_loss = float('inf')
for epoch in tqdm(range(1, num_epochs+1), desc="Training Progress"):
    model.train()
    total_train_loss = 0
    for x, y in tqdm(train_loader, desc="training"):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        pred = model(x)
        loss = criterion(pred, y)
        loss.backward()
        optimizer.step()
        total_train_loss += loss.item() * x.size(0)
    avg_train_loss = total_train_loss / len(train_loader.dataset)

    model.eval()
    total_val_loss = 0
    with torch.no_grad():
        for x, y in tqdm(val_loader, desc="validation"):
            x, y = x.to(device), y.to(device)
            pred = model(x)
            loss = criterion(pred, y)
            total_val_loss += loss.item() * x.size(0)
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
    print(f"[Epoch {epoch}/{num_epochs}] Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}")

# Test
checkpoint = torch.load('best_seaice_gru.pth', map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

test_preds, test_trues = [], []
test_loss = 0
with torch.no_grad():
    for x, y in test_loader:
        x, y = x.to(device), y.to(device)
        pred = model(x)
        loss = criterion(pred, y)
        test_loss += loss.item() * x.size(0)
        test_preds.append(pred.cpu().numpy())
        test_trues.append(y.cpu().numpy())
avg_test_loss = test_loss / len(test_loader.dataset)
print(f"Average Test Loss: {avg_test_loss:.6f}")

# Visualization
def plot_sic_comparison(pred, true, idx, save_path=None):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    im0 = axes[0].imshow(np.squeeze(true[0]), vmin=0, vmax=1, cmap='Blues')
    axes[0].set_title(f"Ground Truth (Idx {idx})")
    axes[0].axis('off')
    plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
    im1 = axes[1].imshow(np.squeeze(pred[0]), vmin=0, vmax=1, cmap='Blues')
    axes[1].set_title(f"Prediction (Idx {idx})")
    axes[1].axis('off')
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    else:
        plt.show()
    plt.close()

# Show sample prediction
sample_idx = 0
plot_sic_comparison(pred=test_preds[sample_idx], true=test_trues[sample_idx], idx=sample_idx)
