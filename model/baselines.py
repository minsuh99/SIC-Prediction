import torch
import torch.nn as nn
import torch.nn.functional as F
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
    
# GRU-based Model
class SeaIceGRU(nn.Module):
    def __init__(self, input_channels=10, hidden_size=64, height=428, width=300, pred_L=1):
        super(SeaIceGRU, self).__init__()
        self.height = height
        self.width = width
        self.pred_L = pred_L
        self.hidden_size = hidden_size
        self.input_size = input_channels * height * width

        self.gru = nn.GRU(input_size=self.input_size, hidden_size=self.hidden_size, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(self.hidden_size, 1024),
            nn.ReLU(),
            nn.Linear(1024, pred_L * height * width)
        )

    def forward(self, x):
        B, L, C, H, W = x.shape
        x = x.view(B, L, -1)
        gru_out, _ = self.gru(x)
        last_output = gru_out[:, -1, :]
        out = self.fc(last_output)
        return out.view(B, self.pred_L, H, W)

# LSTM-based Model
class SeaIceLSTM(nn.Module):
    def __init__(self, input_channels=10, hidden_size=64, height=428, width=300, pred_L=1):
        super(SeaIceLSTM, self).__init__()
        self.height = height
        self.width = width
        self.pred_L = pred_L
        self.hidden_size = hidden_size
        self.input_size = input_channels * height * width

        self.lstm = nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_size, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(self.hidden_size, 1024),
            nn.ReLU(),
            nn.Linear(1024, pred_L * height * width)
        )

    def forward(self, x):
        B, L, C, H, W = x.shape
        x = x.view(B, L, -1)
        lstm_out, _ = self.lstm(x)
        last_output = lstm_out[:, -1, :]
        out = self.fc(last_output)
        return out.view(B, self.pred_L, H, W)
    
# Transformer-based Model
class SeaIceTransformer(nn.Module):
    def __init__(self, input_channels=10, height=428, width=300, d_model=512, nhead=8, num_layers=4, pred_L=1):
        super().__init__()
        self.height = height
        self.width = width
        self.pred_L = pred_L
        self.input_size = input_channels * height * width

        self.input_fc = nn.Linear(self.input_size, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_fc = nn.Sequential(
            nn.Linear(d_model, 1024),
            nn.ReLU(),
            nn.Linear(1024, pred_L * height * width)
        )

    def forward(self, x):
        B, L, C, H, W = x.shape
        x = x.view(B, L, -1)  # (B, L, C*H*W)
        x = self.input_fc(x)  # (B, L, d_model)
        x = self.transformer_encoder(x)  # (B, L, d_model)
        out = x[:, -1, :]  # use the final timestep representation
        out = self.output_fc(out)  # (B, pred_L * H * W)
        return out.view(B, self.pred_L, H, W)  # (B, pred_L, H, W)