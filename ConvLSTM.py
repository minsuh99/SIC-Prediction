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

## Custom Sea Ice Dataset
class SeaIceDataset(Dataset):
    def __init__(self, climate_array, sic_array, window_length, start_idx, end_idx):
        self.climate = climate_array
        self.sic = sic_array
        self.L = window_length                 # Sliding window size

        self.start = start_idx + self.L
        self.end = end_idx

    def __len__(self):
        return self.end - self.start + 1

    def __getitem__(self, idx):
        t = self.start + idx

        # Feature Sequence
        seq_climate = self.climate[t-self.L : t]   # (L, 10, 428, 300)
        # Target
        target_sic = self.sic[t] # (428, 300)
        target_sic = np.expand_dims(target_sic, axis=0) # (1, 428, 300)

        seq_climate = torch.from_numpy(seq_climate).float()
        target_sic = torch.from_numpy(target_sic).float()

        return seq_climate, target_sic
    

train_dataset = SeaIceDataset(climate_array=climate_data, sic_array=sic_data, window_length=12, start_idx=0, end_idx=287)
val_dataset = SeaIceDataset(climate_array=climate_data, sic_array=sic_data, window_length=12, start_idx=288, end_idx=323)
test_dataset = SeaIceDataset(climate_array=climate_data, sic_array=sic_data, window_length=12, start_idx=324, end_idx=359)

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=0, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=0, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=0, pin_memory=True)
## Define Model
# Code by GPT to reproduce the paper
class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, bias=True):
        super(ConvLSTMCell, self).__init__()
        self.input_dim  = input_dim
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

    def forward(self, x, h_prev, c_prev):
        # 1) 입력 x와 이전 은닉 h_prev를 채널 차원으로 concatenate
        combined = torch.cat([x, h_prev], dim=1)  # (batch, input_dim+hidden_dim, H, W)

        # 2) 한 번의 Conv 연산으로 4가지 게이트에 필요한 연산량을 모두 계산
        #    출력 채널 수는 4*hidden_dim
        combined_conv = self.conv(combined)  # (batch, 4*hidden_dim, H, W)

        # 3) 채널을 4개 그룹으로 분리
        cc_i, cc_f, cc_g, cc_o = torch.split(
            combined_conv, self.hidden_dim, dim=1
        )
        # cc_i: 입력 게이트 전 단계 연산 결과
        # cc_f: 망각 게이트 전 단계 연산 결과
        # cc_g: 신규 셀 상태 제안치 전 단계 연산 결과
        # cc_o: 출력 게이트 전 단계 연산 결과

        # 4) 활성화 함수 적용
        i = torch.sigmoid(cc_i)             # 입력 게이트
        f = torch.sigmoid(cc_f)             # 망각 게이트
        g = torch.tanh(cc_g)                # 셀 상태 후보
        o = torch.sigmoid(cc_o)             # 출력 게이트

        # 5) 새로운 셀 상태 계산
        c_next = f * c_prev + i * g

        # 6) 새로운 은닉 상태 계산
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

class ConvLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dims, kernel_size, num_layers, batch_first=False, bias=True, return_all_layers=False):
        super(ConvLSTM, self).__init__()
        self.input_dim  = input_dim
        self.hidden_dims = hidden_dims
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bias = bias
        self.return_all_layers = return_all_layers

        layers = []
        for i in range(self.num_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dims[i-1]
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
            x = x.permute(1, 0, 2, 3, 4) # (L, B, C, H, W) → (B, L, C, H, W)

        batch_size, seq_len, _, height, width = x.size()

        # 초기 은닉 상태가 주어지지 않았다면, 0으로 초기화
        if hidden_states is None:
            hidden_states = []
            for i in range(self.num_layers):
                h = torch.zeros(batch_size, self.hidden_dims[i], height, width, device=x.device)
                c = torch.zeros(batch_size, self.hidden_dims[i], height, width, device=x.device)
                hidden_states.append((h, c))

        layer_output_list = []
        last_state_list   = []

        cur_input = x  # 첫 레이어의 입력은 전체 시퀀스

        # 레이어마다 순차 삽입
        for layer_idx in range(self.num_layers):
            h, c = hidden_states[layer_idx]
            output_inner = []  # 이 레이어 내에서 각 시점별 은닉 상태를 담을 리스트

            # 각 시점(time-step)마다 ConvLSTMCell을 호출
            for t in range(seq_len):
                # t번째 time-step 입력 (batch, C, H, W)
                x_t = cur_input[:, t, :, :, :]
                h, c = self.cell_list[layer_idx](x_t, h, c)
                output_inner.append(h)  # 은닉 상태만 저장 (c는 내부에서 관리)

            # output_inner: 길이 L인 은닉 상태 텐서 리스트. (각각 (B, hidden_dim, H, W))
            # 이를 (B, L, hidden_dim, H, W) 텐서로 합침
            layer_output = torch.stack(output_inner, dim=1)
            layer_output_list.append(layer_output)

            # 해당 레이어의 마지막 은닉 상태와 셀 상태를 저장
            last_state_list.append((h, c))

            # 다음 레이어 입력: 현재 레이어의 전 시퀀스 은닉 출력
            cur_input = layer_output

        # return:
        #  - layer_output_list: [레이어1 전체 시퀀스 출력, 레이어2 전체 시퀀스 출력, ...]
        #  - last_state_list:   [(h_n1, c_n1), (h_n2, c_n2), ...]
        if not self.return_all_layers:
            layer_output_list = layer_output_list[-1:]  # 마지막 레이어만 남김
            last_state_list   = last_state_list[-1:]

        return layer_output_list, last_state_list

class SeaIceConvLSTM(nn.Module):
    def __init__(self, input_channels=10, hidden_channels=64, kernel_size=(3, 3), lstm_layers=1, height=428, width=300, bias=True):
        super(SeaIceConvLSTM, self).__init__()

        # 1) ConvLSTM 모듈
        self.convlstm = ConvLSTM(
            input_dim=input_channels,
            hidden_dims=[hidden_channels] * lstm_layers,
            kernel_size=[kernel_size] * lstm_layers,
            num_layers=lstm_layers,
            batch_first=True,     # 입력 x는 (B, L, C, H, W)
            bias=bias,
            return_all_layers=False
        )

        # 2) 출력 Head: 마지막 은닉 상태 → SIC 예측
        # ConvLSTM의 마지막 레이어 은닉은 (B, 64, H, W)
        self.conv1 = nn.Conv2d(
            in_channels=hidden_channels,
            out_channels=32,
            kernel_size=3,
            padding=1,
            bias=bias
        )
        self.relu  = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            in_channels=32,
            out_channels=1,
            kernel_size=3,
            padding=1,
            bias=bias
        )

    def forward(self, x):
        # 1) ConvLSTM에 시퀀스를 넣으면,
        #    layer_output_list = [(B, L, hidden_channels, H, W)]
        #    last_state_list   = [(h_n, c_n)] 에서
        #    h_n: (B, hidden_channels, H, W)
        layer_output_list, last_state_list = self.convlstm(x)

        # ConvLSTM 마지막 레이어의 마지막 시점 은닉 상태 (h_n)
        h_n, c_n = last_state_list[0]     # 튜플 (h_n, c_n)
        # h_n shape: (B, hidden_channels, 428, 300)

        # 2) 후처리: conv → ReLU → conv
        x = self.conv1(h_n)    # (B, 32, H, W)
        x = self.relu(x)
        out = self.conv2(x)   # (B, 1, H, W)

        return out

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_epochs = 50
model = SeaIceConvLSTM(
    input_channels=10,
    hidden_channels=64,
    kernel_size=(3,3),
    lstm_layers=1,
    height=428,
    width=300,
    bias=True
).to(device)

# Loss & Optimizer & Learning rate Scheduler
criterion = nn.MSELoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)

best_val_loss = float('inf')

## Train & Validation
for epoch in tqdm(range(1, num_epochs+1), desc="Training Progress"):
    # Train
    model.train()
    total_train_loss = 0.0
    for seq_climate, target_sic in tqdm(train_loader, desc="training"):
        seq_climate = seq_climate.to(device) # (B, L, 11, 428, 300)
        target_sic  = target_sic.to(device) # (B, 1, 428, 300)
        
        optimizer.zero_grad()
        pred_sic = model(seq_climate)            # (B, 1, 428, 300)
        loss = criterion(pred_sic, target_sic)
        loss.backward()
        optimizer.step()
        total_train_loss += loss.item() * seq_climate.size(0)

    avg_train_loss = total_train_loss / len(train_loader.dataset)

    # Validation
    model.eval()
    total_val_loss = 0.0
    with torch.no_grad():
        for seq_climate, target_sic in tqdm(val_loader, desc="validation"):
            seq_climate = seq_climate.to(device)
            target_sic = target_sic.to(device)

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
        }, 'best_seaice_convlstm.pth')

    print(f"[Epoch {epoch}/{num_epochs}] Train Loss = {avg_train_loss:.6f}  |  Val Loss = {avg_val_loss:.6f}  |  LR = {optimizer.param_groups[0]['lr']:.2e}")
## Test & Visualization

checkpoint = torch.load('best_seaice_convlstm.pth', map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

test_preds = []
test_trues = []
test_losses = 0.0

with torch.no_grad():
    for seq_climate, target_sic in test_loader:
        seq_climate = seq_climate.to(device)
        target_sic = target_sic.to(device)

        pred_sic = model(seq_climate)
        loss = criterion(pred_sic, target_sic)
        test_losses += loss.item() * seq_climate.size(0)

        test_preds.append(pred_sic.cpu().numpy())
        test_trues.append(target_sic.cpu().numpy())

avg_test_loss = test_losses / len(test_loader.dataset)
print(f"Average Test Loss = {avg_test_loss:.6f}")

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

# 예시: 테스트 10번째 샘플을 시각화
sample_idx = 0
plot_sic_comparison(pred=test_preds[sample_idx], true=test_trues[sample_idx], idx=sample_idx)
