import os
import shutil
import cv2
from tqdm import tqdm
import numpy as np
from collections import defaultdict


def unify_image_resolutions(input_dir, output_dir, target_size=(640, 512)):
    to_return = (0, 0, 1)
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
                to_return = (pad_top, pad_bottom, 1)
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
                to_return = (pad_top, pad_bottom, 0.5)
            else:
                print(
                    f"⚠️ Warning: Unsupported resolution {res} for {img_name}. Skipping."
                )
    print("Processing complete.")
    return to_return


def coco_annotation_adjustment(annotations, pad_top, pad_bottom, coeff=1):
    if pad_top == 0 and pad_bottom == 0 and coeff == 1:
        return annotations
    new_coco = {
        "info": annotations.get("info", {}),
        "licenses": annotations.get("licenses", []),
        "images": [],
        "annotations": [],
        "categories": annotations.get("categories", []),
    }
    # 1. Update Image Dimensions in the JSON metadata
    for img in annotations.get("images", []):
        new_img = img.copy()
        new_img["width"] = 640
        new_img["height"] = 512
        new_coco["images"].append(new_img)

    # 2. Adjust Annotations (Bounding Boxes and Segments)
    for ann in annotations.get("annotations", []):
        new_ann = ann.copy()

        # COCO bbox format: [x_min, y_min, width, height]
        x, y, w, h = ann["bbox"]

        # Apply scaling followed by the top padding offset
        new_x = x * coeff
        new_y = (y * coeff) + pad_top
        new_w = w * coeff
        new_h = h * coeff

        new_ann["bbox"] = [new_x, new_y, new_w, new_h]
        new_ann["area"] = new_w * new_h

        # Handle segmentation polygons if they exist
        if "segmentation" in ann:
            new_seg = []
            for poly in ann["segmentation"]:
                # Flattened list: [x1, y1, x2, y2, ...]
                adjusted_poly = []
                for i in range(0, len(poly), 2):
                    adjusted_poly.append(poly[i] * coeff)  # x
                    adjusted_poly.append((poly[i + 1] * coeff) + pad_top)  # y
                new_seg.append(adjusted_poly)
            new_ann["segmentation"] = new_seg

        new_coco["annotations"].append(new_ann)

    return new_coco


def yolo_annotation_adjustment(
    old_image_path, new_image_path, old_yolo_text_path, new_yolo_text_path
):
    if not os.path.exists(new_yolo_text_path):
        os.makedirs(new_yolo_text_path)

    # Filter for valid images only
    images = [
        f
        for f in os.listdir(old_image_path)
        if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"))
    ]

    for img_name in tqdm(images, desc="Adjusting YOLO Labels"):
        # 1. Setup Paths
        old_img_p = os.path.join(old_image_path, img_name)
        new_img_p = os.path.join(new_image_path, img_name)

        # Robustly handle text file extensions (e.g., image.png -> image.txt)
        base_name = os.path.splitext(img_name)[0]
        old_txt_p = os.path.join(old_yolo_text_path, base_name + ".txt")
        new_txt_p = os.path.join(new_yolo_text_path, base_name + ".txt")

        # Skip if annotation file doesn't exist
        if not os.path.exists(old_txt_p):
            continue

        # 2. Get Dimensions safely
        img_old = cv2.imread(old_img_p)
        img_new = cv2.imread(new_img_p)

        if img_old is None or img_new is None:
            print(f"⚠️ Warning: Could not read image {img_name}. Skipping.")
            continue

        h_old, w_old = img_old.shape[:2]
        h_new, w_new = img_new.shape[:2]

        # 3. Calculate Scale and Padding Dynamically
        # We assume width determines the scale (fitting 1280 or 640 into 640)
        scale = w_new / w_old

        # Calculate what the height would be just after scaling
        scaled_h = h_old * scale

        # The remaining difference is the top padding
        pad_top = (h_new - scaled_h) / 2

        # 4. Process the Annotations
        with open(old_txt_p, "r") as f:
            lines = f.readlines()

        new_lines = []
        for line in lines:
            parts = line.strip().split()
            if len(parts) < 5:
                continue

            cls = int(parts[0])
            x_c, y_c, w, h = map(float, parts[1:])

            # Step A: De-normalize (convert to old absolute pixels)
            abs_x_c = x_c * w_old
            abs_y_c = y_c * h_old
            abs_w = w * w_old
            abs_h = h * h_old

            # Step B: Apply Transformation (Scale -> then Pad)
            new_abs_x_c = abs_x_c * scale
            new_abs_y_c = (abs_y_c * scale) + pad_top
            new_abs_w = abs_w * scale
            new_abs_h = abs_h * scale

            # Step C: Re-normalize (convert to new relative coordinates)
            new_x_c = new_abs_x_c / w_new
            new_y_c = new_abs_y_c / h_new
            new_w = new_abs_w / w_new
            new_h = new_abs_h / h_new

            # Safety Clamp (ensure values stay within 0.0 - 1.0)
            new_x_c = min(max(new_x_c, 0.0), 1.0)
            new_y_c = min(max(new_y_c, 0.0), 1.0)
            new_w = min(max(new_w, 0.0), 1.0)
            new_h = min(max(new_h, 0.0), 1.0)

            new_lines.append(
                f"{cls} {new_x_c:.6f} {new_y_c:.6f} {new_w:.6f} {new_h:.6f}\n"
            )

        # 5. Save Result
        with open(new_txt_p, "w") as f:
            f.writelines(new_lines)
