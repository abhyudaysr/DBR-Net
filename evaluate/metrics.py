import numpy as np
import torch
from skimage.metrics import structural_similarity as ssim_skimage

def calculate_psnr(img1, img2):
    """
    Calculates Peak Signal-to-Noise Ratio (PSNR) between two images.
    Input images must be 3D Tensors (C, H, W).
    """
    
    # 1. Convert Tensors to NumPy Arrays (CRITICAL FIX)
    # Detach, move to CPU, and convert to numpy array for both images
    img1 = img1.detach().cpu().numpy()
    img2 = img2.detach().cpu().numpy()

    # 2. Transpose Axes (C, H, W -> H, W, C) for metric calculation
    # Only transpose if the input is 3D (C, H, W)
    if img1.ndim == 3: 
        img1 = img1.transpose(1, 2, 0)
        img2 = img2.transpose(1, 2, 0)

    # 3. Calculate Mean Squared Error (MSE)
    # np.mean works now because img1 and img2 are guaranteed NumPy arrays
    mse = np.mean((img1 - img2) ** 2)
    
    if mse == 0:
        return 100.0  # PSNR is infinite for identical images
    
    # Max value of the image (assuming normalized input: 0.0 to 1.0)
    PIXEL_MAX = 1.0
    
    # PSNR formula
    return 20 * np.log10(PIXEL_MAX / np.sqrt(mse))

def calculate_ssim(img1, img2):
    """
    Calculates Structural Similarity Index (SSIM) between two images.
    Input images must be 3D Tensors (C, H, W).
    """
    
    # 1. Convert Tensors to NumPy Arrays (CRITICAL FIX)
    img1 = img1.detach().cpu().numpy()
    img2 = img2.detach().cpu().numpy()
    
    # 2. Transpose Axes (C, H, W -> H, W, C) for metric calculation
    if img1.ndim == 3: 
        img1 = img1.transpose(1, 2, 0)
        img2 = img2.transpose(1, 2, 0)
        
    # 3. Calculate SSIM using scikit-image
    # channel_axis=2 is used because the images are now H, W, C
    # data_range=1.0 is safe for normalized inputs (0 to 1)
    return ssim_skimage(img1, img2, data_range=1.0, channel_axis=2, multichannel=True)
