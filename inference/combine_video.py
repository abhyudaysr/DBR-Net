import cv2
import os

# Path to enhanced frames
frames_dir = r"E:\Underwater STEN demo\results\outputs"
output_video = os.path.join(frames_dir, "enhanced_video.mp4")

# Sort frames numerically if they are named like 0001.png, 0002.png, etc.
frames = sorted(
    [f for f in os.listdir(frames_dir) if f.lower().endswith((".png", ".jpg"))],
    key=lambda x: int(''.join(filter(str.isdigit, x)) or 0)
)

if not frames:
    print("❌ No frames found in:", frames_dir)
    exit()

# Read the first frame to get size
first_frame = cv2.imread(os.path.join(frames_dir, frames[0]))
h, w, _ = first_frame.shape

# Create video writer (use mp4v codec)
out = cv2.VideoWriter(output_video, cv2.VideoWriter_fourcc(*'mp4v'), 30, (w, h))

print("🎥 Combining frames into video...")
for f in frames:
    frame_path = os.path.join(frames_dir, f)
    img = cv2.imread(frame_path)
    if img is None:
        print(f"⚠️ Skipping unreadable frame: {f}")
        continue
    out.write(img)

out.release()
print(f"✅ Video saved successfully at: {output_video}")

