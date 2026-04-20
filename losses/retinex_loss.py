# losses/retinex_loss.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.vgg_extractor import VGGExtractor 

class RetinexLoss(nn.Module):
    def __init__(self, lambda_vgg=0.005, lambda_dec=0.01, lambda_smooth=0.1):
        super(RetinexLoss, self).__init__()
        
        # Loss Term Weights (Can be tuned in train.py)
        self.lambda_vgg = lambda_vgg
        self.lambda_dec = lambda_dec
        self.lambda_smooth = lambda_smooth
        
        # 1. Image Reconstruction Loss (L1) - The primary metric
        self.l1_loss = nn.L1Loss()
        
        # 2. Perceptual Loss (VGG)
        self.vgg_extractor = VGGExtractor().cuda()
        self.l2_loss = nn.MSELoss() # L2 is often used for Perceptual Loss

        # 3. Illumination Smoothness Loss (for Retinex)
        self.smooth_loss = self.gradient_loss
        
    def gradient_loss(self, x):
        """Calculates the first-order gradient loss (used for smoothness)"""
        # Calculates difference in horizontal (dx) and vertical (dy) directions
        dh = torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1])
        dw = torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :])
        return torch.mean(dh) + torch.mean(dw)

    def perceptual_loss(self, enhanced_img, target_img):
        """Calculates the distance between feature maps extracted by VGG."""
        enhanced_features = self.vgg_extractor(enhanced_img)
        target_features = self.vgg_extractor(target_img)
        
        loss = 0
        for e_feat, t_feat in zip(enhanced_features, target_features):
            loss += self.l2_loss(e_feat, t_feat)
        return loss

    def forward(self, I_restored, R_prime, L_prime, target_img, degraded_img):
        """
        Calculates the full composite loss.
        I_restored = R' * L' (the enhanced image)
        """
        
        # 1. Image Reconstruction Loss (L1)
        L_img = self.l1_loss(I_restored, target_img)
        
        # 2. Perceptual Loss (L_VGG)
        L_vgg = self.perceptual_loss(I_restored, target_img) * self.lambda_vgg
        
        # 3. Decomposition Loss (L_Dec): Retinex constraint
        # The reconstructed R' * L' should also be close to the degraded input (S=R*L)
        # This ensures the decomposition is valid.
        L_dec = self.l1_loss(R_prime * L_prime, degraded_img) * self.lambda_dec
        
        # 4. Illumination Smoothness Loss (L_Smooth):
        # Enforces the illumination map (L') to be smooth, preventing texture leaks.
        L_smooth = self.smooth_loss(L_prime) * self.lambda_smooth
        
        # Final Total Loss
        L_total = L_img + L_vgg + L_dec + L_smooth
        
        # Return total loss and individual components for monitoring
        return L_total, L_img, L_vgg, L_dec, L_smooth

