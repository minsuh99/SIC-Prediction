import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
import numpy as np
from tqdm.auto import tqdm
from utils.utils import set_seed, visualize_pred_L_1, visualize_pred_L_3, visualize_pred_L_6

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataset import SeaIceDataset, SICOnlyDataset, ClimateSICDataset
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

from model.PNUNet import PNUNet

# Hyperparameter Settings
hyper_params = {
    'window_size': 12,
    'prediction_length': 1,
    'train_start_idx': 0,
    'train_end_idx': 239,
    'val_start_idx': 240,
    'val_end_idx': 299,
    'test_start_idx': 300,
    'test_end_idx': 359,
    
    'batch_size': 4,
    'num_workers': 2,
    'learning_rate':1e-4,
    'epochs': 30,
    
    'tcn_channels': 64,
    'tcn_layers': 3,
    'UNet_features': [2**i for i in range(6, 10)] # [64, 128, 256, 512]

}

# Fix Seed
set_seed(42)

## Data Load
# Feature Data
climate_file = np.load("./preprocessed/ECMWF_climate.npz")
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
cm_mean = climate_train.mean(axis=(0, 2, 3), keepdims=True)
cm_std = climate_train.std(axis=(0, 2, 3), keepdims=True)
climate_data = (climate_data - cm_mean) / (cm_std + 1e-6)

# Define Dataset
train_dataset = SeaIceDataset(climate_array=climate_data, sic_array=sic_data, mask_array=mask, window_length=12, prediction_length=1, start_idx=0, end_idx=239)
val_dataset = SeaIceDataset(climate_array=climate_data, sic_array=sic_data, mask_array=mask, window_length=12, prediction_length=1, start_idx=240, end_idx=299)
test_dataset = SeaIceDataset(climate_array=climate_data, sic_array=sic_data, mask_array=mask, window_length=12, prediction_length=1, start_idx=300, end_idx=359)

# SIC ONLY Dataset 
# train_dataset = SICOnlyDataset(sic_array=sic_data, mask_array=mask, window_length=12, prediction_length=1, start_idx=0, end_idx=239)
# val_dataset = SICOnlyDataset(sic_array=sic_data, mask_array=mask, window_length=12, prediction_length=1, start_idx=240, end_idx=299)
# test_dataset = SICOnlyDataset(sic_array=sic_data, mask_array=mask, window_length=12, prediction_length=1, start_idx=300, end_idx=359)

# Climate and SIC Dataset
# train_dataset = ClimateSICDataset(climate_array=climate_data, sic_array=sic_data, mask_array=mask, window_length=12, prediction_length=1, start_idx=0, end_idx=239)
# val_dataset = ClimateSICDataset(climate_array=climate_data, sic_array=sic_data, mask_array=mask, window_length=12, prediction_length=1, start_idx=240, end_idx=299)
# test_dataset = ClimateSICDataset(climate_array=climate_data, sic_array=sic_data, mask_array=mask, window_length=12, prediction_length=1, start_idx=300, end_idx=359)

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=2, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=2, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=2, pin_memory=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_epochs = 30
model = PNUNet(input_channels=10, tcn_channels=64, tcn_layers=3, UNet_features=[64, 128, 256, 512], pred_L=1).to(device)
# SIC ONLY
# model = PNUNet(input_channels=1, tcn_channels=64, tcn_layers=3, UNet_features=[64, 128, 256, 512], pred_L=1).to(device)
# Climate + SIC
# model = PNUNet(input_channels=11, tcn_channels=64, tcn_layers=3, UNet_features=[64, 128, 256, 512], pred_L=1).to(device)

loss_fn = nn.MSELoss(reduction='none')
optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)
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
        loss_map = loss_fn(pred_sic, target_sic)
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
            loss_map = loss_fn(pred_sic, target_sic)
            loss = (loss_map * mask).sum() / mask.sum()
            total_val_loss += loss.item() * seq_climate.size(0)

    avg_val_loss = total_val_loss / len(val_loader.dataset)

    scheduler.step(avg_val_loss)
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), 'PNUNet.pth')

    tqdm.write(f"[Epoch {epoch}/{num_epochs}] Train Loss = {avg_train_loss:.6f}  |  Val Loss = {avg_val_loss:.6f}  |  LR = {optimizer.param_groups[0]['lr']:.2e}")

## Test & Visualization
model.load_state_dict(torch.load('PNUNet.pth', map_location=device))
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
        loss_map = loss_fn(pred_sic, target_sic)
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
    
# Visualization
test_start_idx = test_dataset.start
test_pred_vis = np.concatenate(test_preds, axis=0)
test_true_vis = np.concatenate(test_trues, axis=0)
test_mask_vis = torch.cat([mask for _, _, mask in test_loader], dim=0).numpy()
save_dir='./results'

# Visualize with prediction length=1
visualize_pred_L_1(pred=test_pred_vis, true=test_true_vis, mask=test_mask_vis, x_coords=x_coords, y_coords=y_coords, dates=dates, start_idx=test_start_idx, save_dir=save_dir)
visualize_pred_L_1(pred=test_pred_vis, true=test_true_vis, mask=test_mask_vis, x_coords=x_coords, y_coords=y_coords, dates=dates, start_idx=test_start_idx, save_dir=save_dir)
visualize_pred_L_1(pred=test_pred_vis, true=test_true_vis, mask=test_mask_vis, x_coords=x_coords, y_coords=y_coords, dates=dates, start_idx=test_start_idx, save_dir=save_dir)

