import torch
import torch.nn as nn
import torch.nn.functional as F

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

class STUNet2D(nn.Module):
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

class SeaIceSTSTUNet(nn.Module):
    def __init__(self, input_channels=10, tcn_channels=64, tcn_layers=3, STunet_features=[64, 128, 256, 512], pred_L=1):
        super().__init__()
        self.pred_L = pred_L
        
        layers = []
        ch = input_channels
        for i in range(tcn_layers):
            layers.append(TCNBlock(ch, tcn_channels, kernel_size=3, dilation=2**i))
            ch = tcn_channels
        self.tcn = nn.Sequential(*layers)
        self.STunet = STUNet2D(tcn_channels, pred_L, STunet_features)

    def forward(self, x):
        # x: (B, L, C, H, W) → (B, C, L, H, W)
        x = x.permute(0,2,1,3,4)
        x = self.tcn(x)           # (B, tcn_channels, L, H, W)
        x = x[:,:, -1, :, :]      # last time → (B, tcn_channels, H, W)
        return self.STunet(x)       # (B, pred_L, H, W)