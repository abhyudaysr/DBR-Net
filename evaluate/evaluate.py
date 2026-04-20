import os
import numpy as np
from skimage.io import imread
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

def eval_dir(pred_dir, gt_dir):
    """
    Calculates PSNR and SSIM for images in pred_dir against gt_dir.
    """
    
    # Get list of files
    if not os.path.exists(pred_dir):
        print(f"Error: Prediction directory not found: {pred_dir}")
        return
    if not os.path.exists(gt_dir):
        print(f"Error: Ground Truth directory not found: {gt_dir}")
        return

    # Sort files to ensure we compare the correct pairs
    pred_files = sorted(os.listdir(pred_dir))
    gt_files = sorted(os.listdir(gt_dir))

    psnr_list = []
    ssim_list = []

    print(f"Found {len(pred_files)} images to evaluate...")

    for i, file_name in enumerate(pred_files):
        # 1. FIX: Handle filename mismatches
        # If your output has "clean_" but ground truth doesn't, remove it!
        clean_name = file_name.replace("clean_", "").replace("enhanced_", "")
        
        # Check if the cleaned name exists in ground truth
        if clean_name in gt_files:
            gt_file_name = clean_name
        elif file_name in gt_files:
            gt_file_name = file_name
        else:
            print(f"Warning: No match found for {file_name} (Tried looking for '{clean_name}')")
            continue
            
        # Construct full paths using the CORRECT names
        pred_path = os.path.join(pred_dir, file_name)
        gt_path = os.path.join(gt_dir, gt_file_name)  # Use the matching name found above

        # 2. Load Images
        try:
            pred_img = imread(pred_path)
            gt_img = imread(gt_path)
        except Exception as e:
            print(f"Error reading {file_name}: {e}")
            continue

        # ... (Rest of the code stays the same) ...

    # 4. Final Results
    if len(psnr_list) > 0:
        avg_psnr = np.mean(psnr_list)
        avg_ssim = np.mean(ssim_list)
        
        print("\n" + "="*30)
        print("EVALUATION RESULTS")
        print("="*30)
        print(f"Images Evaluated: {len(psnr_list)}")
        print(f"Average PSNR:     {avg_psnr:.4f} dB")
        print(f"Average SSIM:     {avg_ssim:.4f}")
        print("="*30)
    else:
        print("\n No valid image pairs were evaluated.")

if __name__ == "__main__":
    # --- UPDATE PATHS HERE IF NEEDED ---
    # Path to where your model saved the enhanced images
    prediction_folder = "results/clean_output" 
    
    # Path to the "Perfect" ground truth images
    ground_truth_folder = "data/test" 
    
    eval_dir(prediction_folder, ground_truth_folder)