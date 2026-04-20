import os
import cv2
import torch
import numpy as np
import torchvision.transforms as tfs
from PIL import Image
from models.dbr_net import DBRNet

# --- CONFIGURATION ---
SHOWCASE_FOLDER = "data/showcase"  # Your curated folder
MODEL_PATH = "results/checkpoints/dbrnet_best.pth"

def run_menu_demo():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n--- 🎓 FINAL YEAR PROJECT DEMO (Device: {device}) ---")

    # 1. Load Model
    if not os.path.exists(MODEL_PATH):
        print("❌ Error: Model weights not found! Train first.")
        return
        
    print("⏳ Loading AI Model...", end="")
    model = DBRNet().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    print(" Done!")

    # 2. List Images
    if not os.path.exists(SHOWCASE_FOLDER):
        print(f"❌ Error: Folder '{SHOWCASE_FOLDER}' not found.")
        return

    images = [f for f in os.listdir(SHOWCASE_FOLDER) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    if not images:
        print("❌ No images found in showcase folder.")
        return

    while True:
        print("\n--- 📂 Available Test Images ---")
        for idx, img_name in enumerate(images):
            print(f" [{idx+1}] {img_name}")
        print(" [Q] Quit")

        # 3. User Selection
        choice = input("\n👉 Select an image number to enhance: ").strip()
        
        if choice.lower() == 'q':
            print("Exiting demo. Good luck!")
            break
        
        if not choice.isdigit() or int(choice) < 1 or int(choice) > len(images):
            print("❌ Invalid selection. Try again.")
            continue
            
        # Get selected image
        selected_img_name = images[int(choice) - 1]
        img_path = os.path.join(SHOWCASE_FOLDER, selected_img_name)
        
        # 4. Process
        print(f"✨ Enhancing {selected_img_name}...")
        process_and_show(model, img_path, device)

def process_and_show(model, img_path, device):
    # Load
    original_pil = Image.open(img_path).convert("RGB")
    
    # Resize for display stability (Optional: Limit huge 4k images to 1080p width)
    w, h = original_pil.size
    if w > 1280:
        ratio = 1280 / w
        new_h = int(h * ratio)
        original_pil = original_pil.resize((1280, new_h))

    to_tensor = tfs.ToTensor()
    input_tensor = to_tensor(original_pil).unsqueeze(0).to(device)

    # Inference
    with torch.no_grad():
        output_tensor = model(input_tensor)

    # Convert Input for OpenCV
    input_cv = cv2.cvtColor(np.array(original_pil), cv2.COLOR_RGB2BGR)

    # Convert Output for OpenCV
    out_img = output_tensor.squeeze(0).cpu().permute(1, 2, 0).numpy()
    out_img = np.clip(out_img, 0, 1)
    out_img = (out_img * 255).astype(np.uint8)
    out_img = cv2.cvtColor(out_img, cv2.COLOR_RGB2BGR)
    
    # Resize output to exactly match input (handles minor padding issues)
    h1, w1 = input_cv.shape[:2]
    out_img = cv2.resize(out_img, (w1, h1))

    # Add Text Labels (Examiners love this)
    cv2.putText(input_cv, "Original Input", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2) # Red Text
    cv2.putText(out_img, "AI Enhanced", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)   # Green Text

    # Stack Side-by-Side
    combined_view = np.hstack((input_cv, out_img))

    # Show
    cv2.imshow(f"Result: {os.path.basename(img_path)}", combined_view)
    print("✅ Displaying result. Press any key on the image window to close it.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_menu_demo()