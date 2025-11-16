import cv2, os

def extract_frames(video_path, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    vid = cv2.VideoCapture(video_path)
    count = 0
    while True:
        ret, frame = vid.read()
        if not ret:
            break
        frame = cv2.resize(frame, (256, 256))
        cv2.imwrite(f"{output_folder}/{count:04d}.png", frame)
        count += 1
    vid.release()
    print(f"Extracted {count} frames to {output_folder}")

if __name__ == "__main__":
    extract_frames("./data/sample_video.mp4", "./data/frames")

