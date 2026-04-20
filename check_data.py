import os

# --- Configuration: CHECK BOTH TRAIN AND VAL SETS ---
INPUT_DIR_TRAIN = 'data/train/input'
GT_DIR_TRAIN = 'data/train/ground_truth'
INPUT_DIR_VAL = 'data/val/input'
GT_DIR_VAL = 'data/val/ground_truth'
# ---------------------------------------------------

def check_consistency(dir1, dir2, set_name):
    print(f"\n--- Checking Consistency for {set_name} Set ---")
    
    # Function to get a set of filenames (without extension)
    def get_base_filenames(directory):
        # We strip the extension (.jpg, .png, etc.) to allow for format differences
        return set(os.path.splitext(f)[0] for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f)))

    set1 = get_base_filenames(dir1)
    set2 = get_base_filenames(dir2)

    if len(set1) != len(set2):
        print(f"🛑 COUNT MISMATCH: {dir1} has {len(set1)} files; {dir2} has {len(set2)} files.")

    # Find the missing files
    missing_in_gt = set1 - set2
    missing_in_input = set2 - set1

    if missing_in_gt:
        print(f"\n--- ❌ MISSING IN GROUND_TRUTH ({dir2}) ---")
        for filename in sorted(list(missing_in_gt))[:10]: # Print top 10 missing
            print(f"Missing file: {filename}")
    
    if missing_in_input:
        print(f"\n--- ❌ MISSING IN INPUT ({dir1}) ---")
        for filename in sorted(list(missing_in_input))[:10]: # Print top 10 missing
            print(f"Missing file: {filename}")

    if not missing_in_gt and not missing_in_input and len(set1) == len(set2):
        print(f"🎉 SUCCESS! {dir1} and {dir2} are perfectly consistent with {len(set1)} files each.")
        return True
    return False

# Run checks for both sets
check_consistency(INPUT_DIR_TRAIN, GT_DIR_TRAIN, "Training")
check_consistency(INPUT_DIR_VAL, GT_DIR_VAL, "Validation")
