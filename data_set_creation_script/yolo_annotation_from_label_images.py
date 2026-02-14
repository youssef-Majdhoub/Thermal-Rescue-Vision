import os
import cv2
import numpy as np

mapping_list = ["Background", "Fire Extinguisher", "Backpack", "Hand Drill", "Survivor"]


def create_yolo_annotation(
    label_image_path, output_annotation_path, needed_indexes=[1, 2, 3, 4]
):
    if not (os.path.exists(output_annotation_path)):
        os.makedirs(output_annotation_path)
    label_image = cv2.imread(label_image_path, cv2.IMREAD_GRAYSCALE)
    if label_image is None:
        print(f"Failed to read image: {label_image_path}")
        return
    h, w = label_image.shape
    yolo_lines = []
    # Get unique pixel values (classes) present in this image
    unique_values = np.unique(label_image)
    print("Unique pixel values found:", unique_values)
    for pixel_value in unique_values:
        if pixel_value == 0:
            continue  # Skip background
        class_id = pixel_value - 1  # Assuming pixel values start from 1 for classes
        if class_id not in needed_indexes:
            continue  # Skip classes not in the needed indexes
        # Create a binary mask for the current class.
        class_id = needed_indexes.index(class_id)
        mask = (label_image == pixel_value).astype(np.uint8) * 255
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            x, y, bw, bh = cv2.boundingRect(contour)
            x_center = (x + bw / 2) / w
            y_center = (y + bh / 2) / h
            width = bw / w
            height = bh / h
            yolo_line = (
                f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n"
            )
            yolo_lines.append(yolo_line)
    file_name = os.path.splitext(os.path.basename(label_image_path))[0] + ".txt"
    output_file_path = os.path.join(output_annotation_path, file_name)
    with open(output_file_path, "w") as f:
        f.writelines(yolo_lines)
    parent_path = os.path.dirname(output_annotation_path)
    additional_info_path = os.path.join(parent_path, "obj.names")
    with open(additional_info_path, "w") as f:
        for index in needed_indexes:
            f.write(mapping_list[index] + "\n")


if __name__ == "__main__":
    output_annotation_path = "C:/Users/medbe/OneDrive/Bureau/PFA2026/real_data/PST900_RGBT_Dataset/train/yolo_annotations"
    label_images_folder = "C:/Users/medbe/OneDrive/Bureau/PFA2026/real_data/PST900_RGBT_Dataset/train/labels/32_bag2a_rect_rgb_frame0000000550.png"
    create_yolo_annotation(
        label_images_folder, output_annotation_path, needed_indexes=[4]
    )
