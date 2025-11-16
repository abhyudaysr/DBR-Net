import torch
import torch.nn as nn
import torch.optim as optim
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from torch.utils.data import DataLoader
from models.sten_lite import STEN_Lite
from training.utils import SimpleFrameDataset, save_checkpoint


def train_demo():
    # Define device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"✅ Using device: {device}")

 
    # Dataset paths
    input_path = './data/train/input'
    target_path = './data/train/target'

    batch_size = 2
    lr = 1e-4
    epochs = 20
    checkpoint_path = './results/checkpoints/demo.pth'

    # Load dataset
    ds = SimpleFrameDataset(input_path, target_path, size=(256, 256))
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True)

    # Initialize model, optimizer, loss
    model = STEN_Lite().to(device)
    optimizer = optim.Adam(model.parameters(), lr=3e-4)

    # StepLR scheduler - reduces LR every few epochs
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)

    criterion = nn.MSELoss()

    print(f"Initial Learning Rate: {optimizer.param_groups[0]['lr']}")

    # ✅ For mixed precision
    scaler = torch.cuda.amp.GradScaler()

    print(f"✅ Model running on: {next(model.parameters()).device}")

    # Training loop
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for imgs, gt in dl:
            imgs = imgs.to(device)
            gt = gt.to(device)

            optimizer.zero_grad()

            # ✅ Initialize ConvLSTM state
            state = model.convlstm.init_state(imgs.size(0), (imgs.shape[2], imgs.shape[3]), device)

            # ✅ Mixed precision forward + backward
            with torch.cuda.amp.autocast():
                out = model(imgs, state)
                if isinstance(out, tuple):
                    out = out[0]
                out = torch.clamp(out, 0, 1)
                loss = criterion(out, gt)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()

        avg_loss = running_loss / len(dl)
        print(f"[Epoch {epoch+1}/{epochs}] Loss: {avg_loss:.4f}")

        scheduler.step()  # Add this here
        save_checkpoint(model, optimizer, epoch, checkpoint_path)

        save_checkpoint(model, optimizer, epoch, checkpoint_path)

    print(f"🎯 Training complete. Model saved to {checkpoint_path}")


if __name__ == "__main__":
    train_demo()
