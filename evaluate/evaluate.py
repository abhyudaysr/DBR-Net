from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import cv2, os
import numpy as np

def eval_dir(pred_dir, gt_dir):
    preds = sorted([os.path.join(pred_dir, f) for f in os.listdir(pred_dir)])
    gts = sorted([os.path.join(gt_dir, f) for f in os.listdir(gt_dir)])
    assert len(preds) == len(gts), "Predictions and ground truth count mismatch!"
    psnr_vals, ssim_vals = [], []
    for p, g in zip(preds, gts):
        ap = cv2.imread(p).astype('float32') / 255.0
        ag = cv2.imread(g).astype('float32') / 255.0
        psnr_vals.append(peak_signal_noise_ratio(ag, ap, data_range=1.0))
        ssim_vals.append(structural_similarity(ag, ap, channel_axis=2))
    return {'PSNR': np.mean(psnr_vals), 'SSIM': np.mean(ssim_vals)}

if __name__ == "__main__":
    results = eval_dir("./results/outputs", "./data/gt_frames")
    print(results)

