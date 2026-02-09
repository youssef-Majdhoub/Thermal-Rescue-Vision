import os
import cv2  # OpenCV for video handling
from PIL import Image
from collections import Counter
from tqdm import tqdm


def get_media_sizes(directory=None):
    # 1. Ask user for the path at runtime if not provided
    if directory is None:
        directory = input("Paste the folder path to scan: ").strip().replace('"', "")

    if not os.path.exists(directory):
        print("❌ Error: That path does not exist.")
        return

    print(f"\n📂 Scanning recursively in: {directory}...")

    image_sizes = []
    video_sizes = []

    # Extensions to look for
    img_exts = (".jpg", ".jpeg", ".png", ".tiff", ".bmp", ".tif")
    vid_exts = (".mp4", ".avi", ".mov", ".mkv", ".flv")

    # 2. Gather all file paths first (Recursive walk)
    all_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            all_files.append(os.path.join(root, file))

    if not all_files:
        print("⚠️ Folder is empty.")
        return

    print(f"found {len(all_files)} files. Analyzing...")

    # 3. Process files with Progress Bar
    for filepath in tqdm(all_files, unit="file"):
        ext = os.path.splitext(filepath)[1].lower()

        # --- IMAGE LOGIC ---
        if ext in img_exts:
            try:
                with Image.open(filepath) as img:
                    image_sizes.append(img.size)
            except:
                pass  # Skip broken images

        # --- VIDEO LOGIC (New!) ---
        elif ext in vid_exts:
            try:
                cap = cv2.VideoCapture(filepath)
                if cap.isOpened():
                    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    video_sizes.append((w, h))
                cap.release()
            except:
                pass  # Skip broken videos

    # 4. Print Report
    print("\n" + "=" * 30)
    print("      SCAN REPORT")
    print("=" * 30)

    # Images
    if image_sizes:
        print(f"\n📸 IMAGES ({len(image_sizes)} found):")
        for size, count in Counter(image_sizes).items():
            print(f"   {size[0]} x {size[1]}  |  Count: {count}")
    else:
        print("\n📸 IMAGES: None found.")

    # Videos
    if video_sizes:
        print(f"\n🎥 VIDEOS ({len(video_sizes)} found):")
        for size, count in Counter(video_sizes).items():
            print(f"   {size[0]} x {size[1]}  |  Count: {count}")
    else:
        print("\n🎥 VIDEOS: None found.")


data_paths = [
    "C:/Users/medbe/OneDrive/Bureau/PFA2026/falling humans/uncommpressed_data/Fall1",
    "C:/Users/medbe/OneDrive/Bureau/PFA2026/archive/FLIR_ADAS_v2/images_thermal_train/data",
    "C:/Users/medbe/OneDrive/Bureau/PFA2026/real_data/PST900_RGBT_Dataset/train/thermal",
]
if __name__ == "__main__":
    for path in data_paths:
        get_media_sizes(path)
