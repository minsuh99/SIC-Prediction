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

## Custom Sea Ice Dataset
class SeaIceDataset(Dataset):
    def __init__(self, climate_array, sic_array, mask_array, window_length, prediction_length, start_idx, end_idx):
        self.climate = climate_array
        self.sic = sic_array
        self.mask = mask_array
        self.L = window_length                 # Sliding window size
        self.pred_L = prediction_length         # e.g. 3 or 6

        # only go up to `len - prediction_length`
        self.start = start_idx + self.L
        self.end = end_idx - (self.pred_L - 1)

    def __len__(self):
        return self.end - self.start + 1

    def __getitem__(self, idx):
        t = self.start + idx

        # Feature Sequence
        seq_climate = self.climate[t-self.L : t]   # (L, 10, 428, 300)
        # Target
        target_sic = self.sic[t : t + self.pred_L] # (pred_L, 428, 300)
        # Mask
        mask = self.mask[t : t + self.pred_L]         # (pred_L, 428, 300)
        
        seq_climate = torch.from_numpy(seq_climate).float()
        target_sic = torch.from_numpy(target_sic).float()
        mask = torch.from_numpy(mask).float()
        valid_mask = 1.0 - mask

        return seq_climate, target_sic, valid_mask

train_dataset = SeaIceDataset(climate_array=climate_data, sic_array=sic_data, mask_array=mask, window_length=12, prediction_length=1, start_idx=0, end_idx=239)
val_dataset = SeaIceDataset(climate_array=climate_data, sic_array=sic_data, mask_array=mask, window_length=12, prediction_length=1, start_idx=240, end_idx=299)
test_dataset = SeaIceDataset(climate_array=climate_data, sic_array=sic_data, mask_array=mask, window_length=12, prediction_length=1, start_idx=300, end_idx=359)

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=2, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=2, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=2, pin_memory=True)
## Define Model
# Code by GPT to reproduce the paper
class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, bias=True):
        super(ConvLSTMCell, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias

        self.conv = nn.Conv2d(
            in_channels=self.input_dim + self.hidden_dim,
            out_channels=4 * self.hidden_dim,
            kernel_size=self.kernel_size,
            padding=self.padding,
            bias=self.bias
        )
        self.dropout = nn.Dropout2d(0.2)

    def forward(self, x, h_prev, c_prev):
        combined = torch.cat([x, h_prev], dim=1)
        combined_conv = self.conv(combined)
        combined_conv = self.dropout(combined_conv)
        cc_i, cc_f, cc_g, cc_o = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        g = torch.tanh(cc_g)
        o = torch.sigmoid(cc_o)
        c_next = f * c_prev + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next

class ConvLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dims, kernel_size, num_layers, batch_first=False, bias=True, return_all_layers=False):
        super(ConvLSTM, self).__init__()
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bias = bias
        self.return_all_layers = return_all_layers

        layers = []
        for i in range(num_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dims[i - 1]
            layers.append(
                ConvLSTMCell(
                    input_dim=cur_input_dim,
                    hidden_dim=self.hidden_dims[i],
                    kernel_size=self.kernel_size[i],
                    bias=self.bias
                )
            )
        self.cell_list = nn.ModuleList(layers)

    def forward(self, x, hidden_states=None):
        if not self.batch_first:
            x = x.permute(1, 0, 2, 3, 4)  # (L, B, C, H, W) → (B, L, C, H, W)

        batch_size, seq_len, _, height, width = x.size()

        if hidden_states is None:
            hidden_states = []
            for i in range(self.num_layers):
                h = torch.zeros(batch_size, self.hidden_dims[i], height, width, device=x.device)
                c = torch.zeros(batch_size, self.hidden_dims[i], height, width, device=x.device)
                hidden_states.append((h, c))

        layer_output_list = []
        last_state_list = []

        cur_input = x
        for layer_idx in range(self.num_layers):
            h, c = hidden_states[layer_idx]
            output_inner = []
            for t in range(seq_len):
                x_t = cur_input[:, t, :, :, :]
                h, c = self.cell_list[layer_idx](x_t, h, c)
                output_inner.append(h)
            layer_output = torch.stack(output_inner, dim=1)
            layer_output_list.append(layer_output)
            last_state_list.append((h, c))
            cur_input = layer_output

        if not self.return_all_layers:
            layer_output_list = layer_output_list[-1:]
            last_state_list = last_state_list[-1:]

        return layer_output_list, last_state_list

class SeaIceConvLSTM(nn.Module):
    def __init__(self, input_channels=10, hidden_channels=64, kernel_size=(3, 3), lstm_layers=1, pred_L=1, bias=True):
        super(SeaIceConvLSTM, self).__init__()

        self.pred_L = pred_L

        self.convlstm = ConvLSTM(
            input_dim=input_channels,
            hidden_dims=[hidden_channels] * lstm_layers,
            kernel_size=[kernel_size] * lstm_layers,
            num_layers=lstm_layers,
            batch_first=True,
            bias=bias,
            return_all_layers=False
        )

        self.conv_head = nn.Sequential(
            nn.Conv2d(hidden_channels, 32, kernel_size=3, padding=1, bias=bias),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.2),
            nn.Conv2d(32, 1, kernel_size=3, padding=1, bias=bias)
        )

    def forward(self, x):
        layer_output_list, _ = self.convlstm(x)  # output shape: (B, L, H, W)
        hidden_seq = layer_output_list[0]        # shape: (B, L, C, H, W)

        last_outputs = hidden_seq[:, -self.pred_L:, :, :, :]  # (B, pred_L, C, H, W)

        preds = []
        for t in range(self.pred_L):
            h_t = last_outputs[:, t]  # (B, C, H, W)
            pred_t = self.conv_head(h_t)  # (B, 1, H, W)
            preds.append(pred_t)

        out = torch.stack(preds, dim=1)  # (B, pred_L, 1, H, W)
        return out.squeeze(2)            # (B, pred_L, H, W)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_epochs = 30
model = SeaIceConvLSTM(input_channels=10, hidden_channels=64, kernel_size=(3,3), lstm_layers=1, pred_L=1, bias=True).to(device)

# Loss & Optimizer & Learning rate Scheduler
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
        torch.save(model.state_dict(),'best_seaice_convlstm_1.pth')
        #### torch.save(model.state_dict(),'best_seaice_Transformer_3.pth')

    tqdm.write(f"[Epoch {epoch}/{num_epochs}] Train Loss = {avg_train_loss:.6f}  |  Val Loss = {avg_val_loss:.6f}  |  LR = {optimizer.param_groups[0]['lr']:.2e}")
## Test & Visualization
model.load_state_dict(torch.load('best_seaice_convlstm_1.pth', map_location=device))
# 위 두줄: model.load_state_dict(torch.load('best_seaice_Transformer_3.pth', map_location=device))

model.eval()

#### 여기부터
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
## 여기까지

# Visualization
def plot_sic_error_map(pred, true, mask, x_coords, y_coords, dates, start_idx, save_dir='./results/ConvLSTM_1'):
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
        norm = matplotlib.colors.BoundaryNorm(boundaries=bounds, ncolors=256)

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

    
# Print Visualization
L = test_dataset.L
test_start_idx = test_dataset.start
test_pred_vis = np.concatenate(test_preds, axis=0)
test_true_vis = np.concatenate(test_trues, axis=0)
test_mask_vis = torch.cat([m for _, _, m in test_loader], dim=0).numpy()

# Visualize in Pred_L(1 months)
plot_sic_error_map(pred=test_pred_vis, true=test_true_vis, mask=test_mask_vis, x_coords=x_coords, y_coords=y_coords, dates=dates, start_idx=test_start_idx)
