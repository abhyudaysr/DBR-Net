import cv2
import os
import numpy as np

input_dir = './data/input_frames'
gt_dir = './data/gt_frames'

for filename in os.listdir(gt_dir):
    gt_path = os.path.join(gt_dir, filename)
    img = cv2.imread(gt_path)

    if img is None:
        continue

    # Simulate underwater degradation: blur + color shift + reduced contrast
    degraded = cv2.GaussianBlur(img, (7, 7), 2)
    degraded = degraded.astype(np.float32)
    degraded[:, :, 1] *= 0.9   # reduce green
    degraded[:, :, 2] *= 0.7   # reduce red → bluish tone
    degraded = np.clip(degraded, 0, 255).astype(np.uint8)

    out_path = os.path.join(input_dir, filename)
    cv2.imwrite(out_path, degraded)

print("Demo degraded frames generated in ./data/input_frames")

