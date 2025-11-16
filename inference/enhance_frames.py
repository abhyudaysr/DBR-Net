import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
print("✅ Project root added to sys.path:", os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import cv2
import torch
import numpy as np
from models.sten_lite import STEN_Lite

# Initialize model
model = STEN_Lite()
 # or your model file name


# Paths

input_dir = './data/train/input'
output_dir = './results/outputs'
model_path = './results/checkpoints/demo.pth'

os.makedirs(output_dir, exist_ok=True)


# Device setup

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# Load trained model

model = STEN_Lite()  # use the same model used for training

# Load trained weights
checkpoint = torch.load(model_path, map_location=device)
model.load_state_dict(checkpoint["model_state"], strict=False)

model.to(device)
model.eval()

print(f"✅ Model weights loaded successfully (non-strict).")


# Frame Enhancement

for filename in os.listdir(input_dir):
    try:
        print(f"Processing: {filename}")
        img_path = os.path.join(input_dir, filename)
        img = cv2.imread(img_path)

        if img is None:
            print(f"⚠️ Skipping {filename} (cannot read image)")
            continue

        # Convert + preprocess
        if img.dtype != np.uint8:
            img = np.clip(img, 0, 255).astype(np.uint8)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (256, 256))
        img_tensor = torch.from_numpy(img.transpose(2, 0, 1)).float().unsqueeze(0) / 255.0
        img_tensor = img_tensor.to(device)

        # Model inference
        with torch.no_grad():
            enhanced = model(img_tensor)

        # Handle model output
        if isinstance(enhanced, (tuple, list)):
            enhanced = enhanced[0]

        # Convert tensor to NumPy
        enhanced = enhanced.squeeze().detach().cpu().numpy().transpose(1, 2, 0)

        # If model outputs in [-1, 1], rescale to [0, 1]
        enhanced = (enhanced + 1) / 2.0
        enhanced = np.clip(enhanced, 0, 1)

        print(f"{filename} -> model output stats: min={enhanced.min():.4f}, max={enhanced.max():.4f}")

        # Save enhanced frame
        enhanced_uint8 = (enhanced * 255).astype(np.uint8)
        enhanced = cv2.convertScaleAbs(enhanced, alpha=1.2, beta=10)

        enhanced_bgr = cv2.cvtColor(enhanced_uint8, cv2.COLOR_RGB2BGR)
        out_path = os.path.join(output_dir, filename)
        cv2.imwrite(out_path, enhanced_bgr)

    except Exception as e:
        print(f"❌ Failed to process {filename}: {e}")
        continue

print(f"\n✨ Enhancement complete! All processed frames saved in: {output_dir}")
