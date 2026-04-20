import os
import torch
import cv2
import numpy as np
from PIL import Image
import torchvision.transforms as tfs
from models.dbr_net import DBRNet

# --- CONFIGURATION ---
INPUT_FOLDER = "data/test"
OUTPUT_FOLDER = "results/clean_output"  # New folder to avoid mixing files
MODEL_PATH = "results/checkpoints/dbrnet_best.pth"

def run_pipeline():
    # --- DEBUG BANNER ---
    print("\n" + "="*50)
    print("✅✅✅ I AM RUNNING THE CLEAN SCRIPT (NO FILTERS) ✅✅✅")
    print("="*50 + "\n")

    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 1. Load Model
    model = DBRNet().to(device)
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        print("✅ Model Weights Loaded Successfully.")
    else:
        print("❌ Model weights NOT found. Aborting.")
        return
    model.eval()

    # 2. Process Images
    image_files = [f for f in os.listdir(INPUT_FOLDER) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    if not image_files:
        print("❌ No images found in data/test!")
        return

    with torch.no_grad():
        for img_name in image_files:
            img_path = os.path.join(INPUT_FOLDER, img_name)
            print(f"Processing {img_name}...")
            
            # Load Original
            original_pil = Image.open(img_path).convert("RGB")
            to_tensor = tfs.ToTensor()
            input_tensor = to_tensor(original_pil).unsqueeze(0).to(device)
            
            # AI Inference
            output_tensor = model(input_tensor)

            # Convert to OpenCV Format
            output_tensor = output_tensor.squeeze(0).cpu()
            output_img = output_tensor.permute(1, 2, 0).numpy()
            
            # --- PURE OUTPUT ---
            # We strictly clip to 0-1 range.
            # If the model is good, this image should look SOFT, not crispy.
            output_img = np.clip(output_img, 0, 1)
            output_img = (output_img * 255).astype(np.uint8)
            output_img = cv2.cvtColor(output_img, cv2.COLOR_RGB2BGR)
            
            # Save
            save_path = os.path.join(OUTPUT_FOLDER, f"clean_{img_name}")
            cv2.imwrite(save_path, output_img)
            print(f"   -> Saved: {save_path}")

    print("\n✅ Done! Check the folder: 'results/clean_output'")

if __name__ == "__main__":
    run_pipeline()
