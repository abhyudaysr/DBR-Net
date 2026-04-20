import torch
from torch.utils.data import Dataset
from PIL import Image
import os
import torchvision.transforms as T
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import cv2

# Simple dataset loader
class SimpleFrameDataset(Dataset):
    def __init__(self, input_dir, target_dir, size=(256, 256)):
        self.input_dir = input_dir
        self.target_dir = target_dir
        self.size = size
        self.images = sorted(os.listdir(input_dir))
        
        # Normalize to [-1, 1]
        self.transform = T.Compose([
            T.Resize(size),
            T.ToTensor(),
            ])



    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        input_name = self.images[idx]
        input_path = os.path.join(self.input_dir, input_name)
        target_path = os.path.join(self.target_dir, input_name)

        img = Image.open(input_path).convert("RGB")
        gt = Image.open(target_path).convert("RGB")

        return self.transform(img), self.transform(gt) 


# Save checkpoint
def save_checkpoint(model, optimizer, epoch, filename):
    torch.save({
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'epoch': epoch
    }, filename)

# Evaluation metrics
def psnr(img1, img2):
    return peak_signal_noise_ratio(img1, img2, data_range=1.0)

def ssim(img1, img2):
    return structural_similarity(img1, img2, multichannel=True)

