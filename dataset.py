import numpy as np
import torch
from torch.utils.data import Dataset

## Custom Sea Ice Dataset
class SeaIceDataset(Dataset):
    def __init__(self, climate_array, sic_array, mask_array, window_length, prediction_length, start_idx, end_idx):
        self.climate = climate_array
        self.sic = sic_array
        self.mask = mask_array
        self.L = window_length                 # Sliding window size
        self.pred_L = prediction_length         # e.g. 1 or 3 or 6

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
        mask = self.mask[t : t + self.pred_L]      # (pred_L, 428, 300)
        
        seq_climate = torch.from_numpy(seq_climate).float()
        target_sic = torch.from_numpy(target_sic).float()
        mask = torch.from_numpy(mask).float()
        valid_mask = 1.0 - mask

        return seq_climate, target_sic, valid_mask
    
## Custom Sea Ice Dataset (Only Past SIC) [Ablation Study]
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
    
## Custom Sea Ice Dataset (Past Climate + SIC) [Ablation Study]
class ClimateSICDataset(Dataset):
    def __init__(self, climate_array, sic_array, mask_array, window_length, prediction_length, start_idx, end_idx):
        self.climate = climate_array
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
        seq_climate = self.climate[t - self.L : t]       # (L, 10, 428, 300)
        seq_sic = self.sic[t - self.L : t]               # (L, 428, 300)
        target_sic = self.sic[t : t + self.pred_L]       # (pred_L, 428, 300)
        mask = self.mask[t : t + self.pred_L]            # (pred_L, 428, 300)
        
        seq_sic_expanded = np.expand_dims(seq_sic, axis=1)  # (L, 1, 428, 300)
        seq_input = np.concatenate([seq_climate, seq_sic_expanded], axis=1)  # (L, 11, 428, 300)
        
        seq_input = torch.from_numpy(seq_input).float()   # (L, 11, 428, 300)
        target_sic = torch.from_numpy(target_sic).float()
        mask = torch.from_numpy(mask).float()
        valid_mask = 1.0 - mask

        return seq_input, target_sic, valid_mask