import cv2
import os
from tqdm import tqdm

# Paths
frames_dir = './results/outputs'
output_video = './results/outputs/enhanced_video.mp4'
fps =30

# Get frame list
frames = sorted([f for f in os.listdir(frames_dir) if f.endswith('.png') or f.endswith('.jpg')])

if not frames:
    print("⚠️ No frames found in", frames_dir)
    print("Please ensure your enhanced frames are saved before running this.")
    exit()

# Read first frame to get size
first_frame = cv2.imread(os.path.join(frames_dir, frames[0]))
height, width, _ = first_frame.shape

# Define video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
fps = 24
out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

# Write frames to video
print(f"🧩 Creating video from {len(frames)} frames...")
for f in tqdm(frames):
    frame = cv2.imread(os.path.join(frames_dir, f))
    out.write(frame)

out.release()
print(f"✅ Enhanced video saved to {output_video}")
