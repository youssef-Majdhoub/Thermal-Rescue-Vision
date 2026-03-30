import os
from tqdm import tqdm
import shutil
import json
import yaml


# Function to copy images in folder called data
def copy_images_in_data_folder(source_folder, destination_folder):
    files = os.listdir(source_folder)
    for file in tqdm(files, desc=f"Copying images from {source_folder}"):
        source_file = os.path.join(source_folder, file)
        destination_file = os.path.join(destination_folder, file)
        shutil.copy2(source_file, destination_file)


# function to merge coco annotations
def merge_coco_annotations(destination_folder, coco_dicts=[]):
    merged_coco = {
        "info": coco_dicts[0]["info"],
        "licenses": coco_dicts[0]["licenses"],
        "categories": coco_dicts[0]["categories"],
        "images": [],
        "annotations": [],
    }
    annotation_id = 1
    add_image_id = 1
    for coco_dict in coco_dicts:
        for image in coco_dict["images"]:
            merged_coco["images"].append(image)
            merged_coco["images"][-1]["id"] += add_image_id
        for annotation in coco_dict["annotations"]:
            annotation["id"] = annotation_id
            annotation["image_id"] += add_image_id
            merged_coco["annotations"].append(annotation)
            annotation_id += 1
        add_image_id += len(coco_dict["images"])
    with open(os.path.join(destination_folder, "coco.json"), "w") as f:
        json.dump(merged_coco, f, indent=4)


# creating yolo annotations function
def merge_yolo_annotations(destination_folder, yolo_annotation_folders=[]):
    data_path = os.path.join(destination_folder, "data")
    if not os.path.exists(data_path):
        os.makedirs(data_path)
    list_to_remove = os.listdir(data_path)
    for file in list_to_remove:
        file_path = os.path.join(data_path, file)
        if os.path.isfile(file_path):
            os.remove(file_path)
    with open(os.path.join(destination_folder, "obj.names"), "w") as f:
        names = []
        for folder in yolo_annotation_folders:
            obj_names_path = os.path.join(folder, "obj.names")
            with open(obj_names_path, "r") as obj_file:
                obj_names = obj_file.readlines()
                obj_names = [name.strip() for name in obj_names]
                for name in obj_names:
                    if name not in names:
                        names.append(name)
                        f.write(name + "\n")
    for folder in yolo_annotation_folders:
        # the file names are unique so the yolo annotations can be merged by simply copying the .txt files
        data_folder = os.path.join(folder, "data")
        for file in tqdm(
            os.listdir(data_folder), desc=f"Copying annotations from {folder}"
        ):
            source_file = os.path.join(data_folder, file)
            destination_file = os.path.join(data_path, file)
            shutil.copy2(source_file, destination_file)
    # creating a yaml file for the merged dataset
    yaml_data = {
        "train": os.path.join(destination_folder, "data"),
        "val": os.path.join(destination_folder, "data"),
        "nc": len(names),
        "names": names,
    }
    with open(os.path.join(destination_folder, "data.yaml"), "w") as yaml_file:
        yaml.dump(yaml_data, yaml_file, default_flow_style=False)


def create_final_training_set(
    destination_folder, flir_folder, falling_human_folder, pst900_folder
):
    final_training_folder = os.path.join(destination_folder, "training")
    os.makedirs(final_training_folder, exist_ok=True)
    data_path = os.path.join(final_training_folder, "data")
    if not os.path.exists(data_path):
        os.makedirs(data_path)
    yolo_path = os.path.join(final_training_folder, "yolo_annotations")
    if not os.path.exists(yolo_path):
        os.makedirs(yolo_path)
    training_folders = [
        os.path.join(flir_folder, "training"),
        os.path.join(falling_human_folder, "training"),
        os.path.join(pst900_folder, "training"),
    ]
    coco_dicts = []
    yolo_paths = []
    for folder in training_folders:
        data_source = os.path.join(folder, "data")
        copy_images_in_data_folder(data_source, data_path)
        coco_path = os.path.join(folder, "coco.json")
        with open(coco_path, "r") as coco_file:
            coco_dict = json.load(coco_file)
            coco_dicts.append(coco_dict)
        yolo_source_path = os.path.join(folder, "yolo_annotations")
        yolo_paths.append(yolo_source_path)
    merge_coco_annotations(final_training_folder, coco_dicts)
    merge_yolo_annotations(yolo_path, yolo_paths)


def create_final_validation_set(
    destination_folder, flir_folder, falling_human_folder, pst900_folder
):
    final_validation_folder = os.path.join(destination_folder, "validation")
    os.makedirs(final_validation_folder, exist_ok=True)
    data_path = os.path.join(final_validation_folder, "data")
    if not os.path.exists(data_path):
        os.makedirs(data_path)
    yolo_path = os.path.join(final_validation_folder, "yolo_annotations")
    if not os.path.exists(yolo_path):
        os.makedirs(yolo_path)
    validation_folders = [
        os.path.join(flir_folder, "testing"),
        os.path.join(falling_human_folder, "testing"),
        os.path.join(pst900_folder, "testing"),
    ]
    coco_dicts = []
    yolo_paths = []
    for folder in validation_folders:
        data_source = os.path.join(folder, "data")
        copy_images_in_data_folder(data_source, data_path)
        coco_path = os.path.join(folder, "coco.json")
        with open(coco_path, "r") as coco_file:
            coco_dict = json.load(coco_file)
            coco_dicts.append(coco_dict)
        yolo_source_path = os.path.join(folder, "yolo_annotations")
        yolo_paths.append(yolo_source_path)
    merge_coco_annotations(final_validation_folder, coco_dicts)
    merge_yolo_annotations(yolo_path, yolo_paths)


if __name__ == "__main__":
    source_folder = "C:/Users/medbe/OneDrive/Bureau/PFA2026/final_data_sets"
    flir_folder = os.path.join(source_folder, "Flir_data_set")
    pst900_folder = os.path.join(source_folder, "PST900_data_set")
    falling_human_folder = os.path.join(source_folder, "falling_human")
    destination_folder = (
        "C:/Users/medbe/OneDrive/Bureau/PFA2026/final_data_sets/final_data_set"
    )
    os.makedirs(destination_folder, exist_ok=True)
    create_final_training_set(
        destination_folder, flir_folder, falling_human_folder, pst900_folder
    )
    create_final_validation_set(
        destination_folder, flir_folder, falling_human_folder, pst900_folder
    )
