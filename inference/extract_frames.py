import os
import cv2
import numpy as np

input_dir = r"E:\Underwater STEN demo\data\train\input"
output_dir = r"E:\Underwater STEN demo\results\outputs"
os.makedirs(output_dir, exist_ok=True)

print("Enhancing underwater frames (simple mode)...")
print("Current working directory:", os.getcwd())
print("Reading input frames from:", input_dir)


def enhance_image(img):
    # --- 1. Basic brightness and contrast adjustment ---
    alpha = 1.25  # contrast control (1.0–3.0)
    beta = 20     # brightness control (0–100)
    img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

    # --- 2. Increase red channel slightly (to balance blue-green tint) ---
    b, g, r = cv2.split(img)
    r = cv2.add(r, 25)  # add warmth
    img = cv2.merge((b, g, r))

    # --- 3. Apply light sharpening ---
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    img = cv2.filter2D(img, -1, kernel)

    # --- 4. Optional denoising ---
    img = cv2.fastNlMeansDenoisingColored(img, None, 5, 5, 7, 21)

    return img


# --- Process all frames ---
for filename in sorted(os.listdir(input_dir)):
    if not filename.lower().endswith((".png", ".jpg", ".jpeg")):
        continue

    img_path = os.path.join(input_dir, filename)
    img = cv2.imread(img_path)
    if img is None:
        print(f"⚠️ Skipping unreadable frame: {filename}")
        continue

    print(f"{filename} shape: {img.shape}, mean pixel value: {np.mean(img):.2f}")


    enhanced = enhance_image(img)
    out_path = os.path.join(output_dir, filename)
    cv2.imwrite(out_path, enhanced)
    print(f"Processed: {filename}")

print(f"✅ Enhanced frames saved in {output_dir}")
