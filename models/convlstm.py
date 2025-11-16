import torch
import torch.nn as nn

# Simple ConvLSTM cell (single-step use)
class ConvLSTMCell(nn.Module):
    def __init__(self, in_ch, hid_ch, kernel_size=3):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(in_ch + hid_ch, 4 * hid_ch, kernel_size, padding=padding)
        self.hid_ch = hid_ch

    def forward(self, x, prev_state):
        h_prev, c_prev = prev_state

        # Ensure both tensors are on the same device
        device = x.device
        h_prev = h_prev.to(device)
        c_prev = c_prev.to(device)

        combined = torch.cat([x, h_prev], dim=1)
        conv_out = self.conv(combined)
        i, f, o, g = torch.split(conv_out, self.hid_ch, dim=1)

        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        o = torch.sigmoid(o)
        g = torch.tanh(g)

        c = f * c_prev + i * g
        h = o * torch.tanh(c)

        return h, c

    def init_state(self, batch, shape, device):
        h = torch.zeros(batch, self.hid_ch, *shape, device=device)
        c = torch.zeros(batch, self.hid_ch, *shape, device=device)
        return h, c
