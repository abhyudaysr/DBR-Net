# models/dbr_net.py
import torch
import torch.nn as nn

class DBRNet(nn.Module):
    def __init__(self):
        super(DBRNet, self).__init__()
        
        # --- Initial Feature Extraction ---
        # A simple convolution to get started
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        
        # --- Branch 1: Detail Enhancement (Standard Convs) ---
        # Focuses on edges and textures
        self.b1_conv1 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.b1_conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        
        # --- Branch 2: Context/Color (Dilated Convs) ---
        # Dilated convolutions see a wider area (good for color correction)
        self.b2_conv1 = nn.Conv2d(32, 32, kernel_size=3, padding=2, dilation=2)
        self.b2_conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=2, dilation=2)
        
        # --- Fusion Layer ---
        # Combine the 32 features from Branch 1 and 32 from Branch 2
        self.fusion = nn.Conv2d(64, 32, kernel_size=1)
        
        # --- Final Reconstruction ---
        self.final_conv = nn.Conv2d(32, 3, kernel_size=3, padding=1)
        
    def forward(self, x):
        # 1. Extract initial features
        x1 = self.relu(self.conv1(x))
        
        # 2. Run Branch 1
        b1 = self.relu(self.b1_conv1(x1))
        b1 = self.relu(self.b1_conv2(b1))
        
        # 3. Run Branch 2
        b2 = self.relu(self.b2_conv1(x1))
        b2 = self.relu(self.b2_conv2(b2))
        
        # 4. Concatenate (stack them together)
        cat = torch.cat([b1, b2], dim=1)
        
        # 5. Fuse and Residual Connection
        out = self.relu(self.fusion(cat))
        out = self.final_conv(out)
        
        # 6. Add original input (Residual Learning)
        return out + x
