import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchmetrics.functional import peak_signal_noise_ratio as psnr_func
from torchmetrics.functional import structural_similarity_index_measure as ssim_func

# Imports
from models.dbr_net import DBRNet
from datasets.underwater_dataset import UnderwaterDataset 

def train():
    # --- CONFIGURATION ---
    EPOCHS = 100
    BATCH_SIZE = 8
    LR = 1e-4
    IMG_SIZE = 256
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    train_dir = "data/train" 
    
    print(f"--- Starting Training (Target: High PSNR) on {DEVICE} ---")
    
    # 1. Data Setup
    try:
        dataset = UnderwaterDataset(train_dir, size=IMG_SIZE)
        loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
        print(f"Dataset Size: {len(dataset)} images") # Confirms your reduction
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return

    # 2. Model Setup
    model = DBRNet().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR, betas=(0.9, 0.999))
    
    # 3. Loss Functions
    # L1 Loss = Brightness/Color Accuracy (Crucial for high PSNR)
    # SSIM Loss = Structure/Sharpness
    criterion_l1 = nn.L1Loss()
    
    # Check for existing checkpoint
    start_epoch = 0
    if os.path.exists("results/checkpoints/dbrnet_best.pth"):
        print("Loading previous best model...")
        # We add map_location=torch.device('cpu') to fix the error
        model.load_state_dict(torch.load("results/checkpoints/dbrnet_best.pth", map_location=torch.device('cpu')))

    # Tracking Best Metrics
    best_psnr = 0.0
    best_ssim = 0.0

    # 4. Training Loop
    for epoch in range(start_epoch, EPOCHS):
        model.train()
        loop = tqdm(loader, leave=True)
        
        total_loss = 0
        total_psnr = 0
        total_ssim = 0
        
        for inputs, targets in loop:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            
            # --- LOSS CALCULATION ---
            loss_l1 = criterion_l1(outputs, targets)
            loss_ssim_val = 1 - ssim_func(outputs, targets, data_range=1.0)
            
            # Weighted Loss: 85% Color/Brightness, 15% Structure
            loss = (0.85 * loss_l1) + (0.15 * loss_ssim_val)
            
            loss.backward()
            optimizer.step()
            
            # --- METRICS ---
            batch_psnr = psnr_func(outputs, targets, data_range=1.0)
            batch_ssim = ssim_func(outputs, targets, data_range=1.0)
            
            total_loss += loss.item()
            total_psnr += batch_psnr.item()
            total_ssim += batch_ssim.item()
            
            loop.set_description(f"Epoch [{epoch+1}/{EPOCHS}]")
            loop.set_postfix(loss=f"{loss.item():.4f}", psnr=f"{batch_psnr.item():.2f}")

        # End of Epoch Stats
        avg_psnr = total_psnr / len(loader)
        avg_ssim = total_ssim / len(loader)
        
        # Save Best Model
        if avg_psnr > best_psnr:
            best_psnr = avg_psnr
            best_ssim = avg_ssim # Capture SSIM at best PSNR
            os.makedirs("results/checkpoints", exist_ok=True)
            torch.save(model.state_dict(), "results/checkpoints/dbrnet_best.pth")
            print(f"   🌟 New Best PSNR: {best_psnr:.2f} dB | SSIM: {best_ssim:.3f}")

    # --- FINAL SUMMARY ---
    print("\n" + "="*40)
    print(f"TRAINING COMPLETE")
    print(f"Best PSNR Achieved: {best_psnr:.4f} dB")
    print(f"Best SSIM Achieved: {best_ssim:.4f}")
    print("="*40 + "\n")

if __name__ == "__main__":
    train()
