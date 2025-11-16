import torch.nn as nn
from .unet import UNetSmall
from .convlstm import ConvLSTMCell
import torch

class STEN_Lite(nn.Module):
    def __init__(self, device='cuda'):
        super().__init__()
        self.unet = UNetSmall()
        self.convlstm = ConvLSTMCell(in_ch=3, hid_ch=16)
        self.refine = nn.Conv2d(16, 3, 1)
        self.device = device

    def forward(self, x, state=None):
        feat = self.unet(x)  # Spatial enhancement
        if state is None:
            state = self.convlstm.init_state(x.size(0), (x.shape[2], x.shape[3]), torch.device("cpu"))
        h, c = self.convlstm(feat, state)
        out = self.refine(h)
        out = torch.clamp(out, 0., 1.)
        return out, (h, c)

