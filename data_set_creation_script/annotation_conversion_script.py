import json
import cv2
import os


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
    for annotation in coco_data["annotations"]:
        image_id = annotation["image_id"]
        category_id = annotation["category_id"]
        if category_id not in needed_categories:
            continue
        bbox = annotation["bbox"]  # COCO format: [x_min, y_min, width, height]
        image_info = next(
            (img for img in coco_data["images"] if img["id"] == image_id), None
        )
        if image_info is None:
            print(f"Image with ID {image_id} not found in COCO data.")
            continue
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
        output_file_path = os.path.join(data_path, f"{image_id}.txt")
        with open(output_file_path, "w") as f:
            f.write(yolo_line)
