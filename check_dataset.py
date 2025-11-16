import os
from PIL import Image

input_dir = r"E:\Underwater STEN demo\data\train\input"
target_dir = r"E:\Underwater STEN demo\data\train\target"

input_files = sorted(os.listdir(input_dir))
target_files = sorted(os.listdir(target_dir))

print("Input frames:", len(input_files))
print("Target frames:", len(target_files))

# Check one pair visually
sample = input_files[0]
print(f"Sample frame: {sample}")
print(f"Exists in target? {'✅' if sample in target_files else '❌'}")

# Check size match
img_in = Image.open(os.path.join(input_dir, sample))
img_tg = Image.open(os.path.join(target_dir, sample))
print("Input size:", img_in.size, "Target size:", img_tg.size)

