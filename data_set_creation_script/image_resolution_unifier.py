import os
import shutil
import cv2
from tqdm import tqdm
import numpy as np
from collections import defaultdict


def unify_image_resolutions(input_dir, output_dir, target_size=(640, 512)):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    images = [
        f
        for f in os.listdir(input_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png", ".tiff", ".bmp", ".tif"))
    ]
    print(f"Found {len(images)} images. Processing...")
    resolutions = defaultdict(list)
    for img_name in images:
        img_path = os.path.join(input_dir, img_name)
        img = cv2.imread(img_path)

        if img is None:
            print(f"⚠️ Warning: Could not read {img_name}. Skipping.")
            continue

        h, w = img.shape[:2]
        resolutions[(w, h)].append(img_name)
    for res, img_list in resolutions.items():
        print(f"Resolution {res}: {len(img_list)} images.")
        for img_name in tqdm(img_list, desc=f"Processing {res}", unit="image"):
            img_path = os.path.join(input_dir, img_name)
            img = cv2.imread(img_path)

            if img is None:
                print(f"⚠️ Warning: Could not read {img_name}. Skipping.")
                continue
            elif res == (640, 512):
                shutil.copy(img_path, os.path.join(output_dir, img_name))
                continue
            elif res == (640, 480):
                # Pad to 640x512
                pad_top = (512 - 480) // 2
                pad_bottom = 512 - 480 - pad_top
                padded_img = cv2.copyMakeBorder(
                    img, pad_top, pad_bottom, 0, 0, cv2.BORDER_CONSTANT, value=[0, 0, 0]
                )
                cv2.imwrite(os.path.join(output_dir, img_name), padded_img)
                return pad_top, pad_bottom
            elif res == (1280, 720):
                # Resize to 640x360
                resized_img = cv2.resize(img, (640, 360), interpolation=cv2.INTER_AREA)
                # Pad to 640x512
                pad_top = (512 - 360) // 2
                pad_bottom = 512 - 360 - pad_top
                padded_img = cv2.copyMakeBorder(
                    resized_img,
                    pad_top,
                    pad_bottom,
                    0,
                    0,
                    cv2.BORDER_CONSTANT,
                    value=[0, 0, 0],
                )
                cv2.imwrite(os.path.join(output_dir, img_name), padded_img)
                return pad_top, pad_bottom
            else:
                print(
                    f"⚠️ Warning: Unsupported resolution {res} for {img_name}. Skipping."
                )
