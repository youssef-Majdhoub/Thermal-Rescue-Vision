import json
import cv2
import os
from tqdm import tqdm


def convert_coco_to_yolo(
    coco_annotation_path, output_annotation_path, images_path, needed_categories=[]
):
    with open(coco_annotation_path, "r") as f:
        coco_data = json.load(f)
    if not os.path.exists(output_annotation_path):
        os.makedirs(output_annotation_path)
    data_path = os.path.join(output_annotation_path, "data")
    if not os.path.exists(data_path):
        os.makedirs(data_path)
    list_to_remove = os.listdir(data_path)
    for file in list_to_remove:
        file_path = os.path.join(data_path, file)
        if os.path.isfile(file_path):
            os.remove(file_path)
    for annotation in tqdm(coco_data["annotations"], desc="Converting annotations"):
        image_id = annotation["image_id"]
        category_id = annotation["category_id"]
        if category_id not in needed_categories:
            continue
        category_id = needed_categories.index(category_id)
        bbox = annotation["bbox"]  # COCO format: [x_min, y_min, width, height]
        image_info = next(
            (img for img in coco_data["images"] if img["id"] == image_id), None
        )
        image = cv2.imread(os.path.join(images_path, image_info["file_name"]))
        if image is None:
            print(f"Failed to read image: {image_info['file_name']}")
            continue
        image_height, image_width = image.shape[:2]
        x_center = (bbox[0] + bbox[2] / 2) / image_width
        y_center = (bbox[1] + bbox[3] / 2) / image_height
        width = bbox[2] / image_width
        height = bbox[3] / image_height
        yolo_line = (
            f"{category_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n"
        )
        last_name = os.path.splitext(os.path.basename(image_info["file_name"]))[0]
        output_file_path = os.path.join(data_path, f"{last_name}.txt")
        with open(output_file_path, "a") as f:
            f.write(yolo_line)
    object_names_path = os.path.join(output_annotation_path, "obj.names")
    with open(object_names_path, "w") as f:
        for id in needed_categories:
            category_info = next(
                (cat for cat in coco_data["categories"] if cat["id"] == id), None
            )
            if category_info is not None:
                f.write(category_info["name"] + "\n")
            else:
                print(f"Category with ID {id} not found in COCO data.")


def convert_yolo_to_coco(
    yolo_annotation_path, images_path, output_json_path, needed_categories=None
):
    """
    Args:
        yolo_annotation_path: Path to folder with .txt files
        images_path: Path to folder with images
        output_json_path: Where to save the result .json
        needed_categories: List of class names (e.g., ["dog", "cat"]).
                           Index 0 is "dog", Index 1 is "cat".
    """

    # 1. Initialize the COCO dictionary structure
    coco_data = {
        "info": {"description": "Converted from YOLO to COCO"},
        "images": [],
        "annotations": [],
        "categories": [],
    }

    # 2. Add Categories (The "Map")
    # If the user provided names, we add them here.
    # COCO needs to know that ID 0 = "cat", ID 1 = "dog", etc.
    categories_path = os.path.join(yolo_annotation_path, "obj.names")
    with open(categories_path, "r") as f:
        category_names = [line.strip() for line in f.readlines()]
    if needed_categories is None:
        pass
    else:
        category_names = [category_names[i] for i in needed_categories]
    for index, name in enumerate(category_names):
        coco_data["categories"].append(
            {"id": index, "name": name, "supercategory": "none"}
        )

    # Global counters for unique IDs
    annotation_id = 1
    image_id = 1

    # Get all .txt files
    yolo_files = [f for f in os.listdir(yolo_annotation_path) if f.endswith(".txt")]

    for yolo_file in tqdm(yolo_files, desc="Converting"):

        # 3. Find the matching image
        # YOLO file is "image_01.txt", we need "image_01.jpg" or ".png"
        base_name = os.path.splitext(yolo_file)[0]
        image_name = None

        # Check common extensions
        for ext in [".jpg", ".jpeg", ".png", ".bmp"]:
            if os.path.exists(os.path.join(images_path, base_name + ext)):
                image_name = base_name + ext
                break

        if image_name is None:
            print(f"Warning: Image not found for {yolo_file}, skipping.")
            continue

        # 4. Read Image Size (CRITICAL STEP)
        # YOLO doesn't know image size, but COCO *requires* it.
        # We must open the image to get height/width.
        img = cv2.imread(os.path.join(images_path, image_name))
        if img is None:
            continue

        h_img, w_img = img.shape[:2]

        # Add image info to COCO 'images' list
        coco_data["images"].append(
            {"id": image_id, "file_name": image_name, "width": w_img, "height": h_img}
        )

        # 5. Read Annotations from .txt
        with open(os.path.join(yolo_annotation_path, yolo_file), "r") as f:
            lines = f.readlines()

        for line in lines:
            parts = line.strip().split()
            # Expecting: class_id x_center y_center width height
            if len(parts) < 5:
                continue

            class_id = int(parts[0])
            x_c_norm = float(parts[1])  # 0.0 to 1.0
            y_c_norm = float(parts[2])  # 0.0 to 1.0
            w_norm = float(parts[3])  # 0.0 to 1.0
            h_norm = float(parts[4])  # 0.0 to 1.0

            # -----------------------------------------------------------
            # THE MATH CONVERSION (YOLO -> COCO)
            # -----------------------------------------------------------

            # A. Un-normalize (Percentage -> Pixels)
            w_px = w_norm * w_img
            h_px = h_norm * h_img
            x_c_px = x_c_norm * w_img
            y_c_px = y_c_norm * h_img

            # B. Center -> Top-Left Corner
            # COCO wants the top-left corner (x_min, y_min), not the center.
            # To get to the left edge, move left by half the width.
            # To get to the top edge, move up by half the height.
            x_min = x_c_px - (w_px / 2)
            y_min = y_c_px - (h_px / 2)

            # Add to 'annotations' list
            coco_data["annotations"].append(
                {
                    "id": annotation_id,
                    "image_id": image_id,  # This links the box to the image above
                    "category_id": class_id,
                    "bbox": [x_min, y_min, w_px, h_px],  # COCO format
                    "area": w_px * h_px,
                    "iscrowd": 0,
                }
            )

            annotation_id += 1  # Increment for next box

        image_id += 1  # Increment for next image

    # 6. Save the final JSON
    with open(output_json_path, "w") as f:
        json.dump(coco_data, f, indent=4)
    print(f"Success! Saved to {output_json_path}")


# --- Example Usage ---
# classes = ["person", "car", "dog"]
# convert_yolo_to_coco("my_yolo_txts/", "my_images/", "output.json", classes)
