import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.net(x)


class UNetSmall(nn.Module):
    def __init__(self, in_ch=3, out_ch=3, features=[32, 64, 128]):
        super().__init__()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()

        prev = in_ch

        # Encoder / Downsampling
        for f in features:
            self.downs.append(DoubleConv(prev, f))
            prev = f

        # Bottleneck
        self.bottleneck = DoubleConv(prev, prev * 2)
        prev = prev * 2  # update prev after bottleneck

        # Decoder / Upsampling (fixed channel mismatch)
        for f in reversed(features):
            self.ups.append(nn.ConvTranspose2d(prev, f, 2, stride=2))
            self.ups.append(DoubleConv(f * 2, f))  # ✅ expects concatenated skip + upsampled
            prev = f

        # Final output layer
        self.final = nn.Conv2d(prev, out_ch, 1)

    def forward(self, x):
        skip = []
        out = x

        # Encoder path
        for d in self.downs:
            out = d(out)
            skip.append(out)
            out = F.max_pool2d(out, 2)

        # Bottleneck
        out = self.bottleneck(out)

        # Decoder path
        for i in range(0, len(self.ups), 2):
            upconv = self.ups[i]
            dconv = self.ups[i + 1]
            out = upconv(out)
            out = torch.cat([out, skip.pop()], dim=1)  # concatenate skip connection
            out = dconv(out)

        # Output layer
        return torch.sigmoid(self.final(out))  # ensures output in [0, 1]
