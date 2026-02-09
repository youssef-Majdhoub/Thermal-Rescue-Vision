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
        output_file_path = os.path.join(
            data_path, f"{os.path.splitext(image_info['file_name'])[0]}.txt"
        )
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
