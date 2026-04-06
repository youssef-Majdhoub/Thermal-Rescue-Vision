import os
from PIL import Image
from download_falling_human import download_falling_human
from download_PST900 import download_PST900
from download_flir_adas import download_flir_adas_dataset
from test_download import download_test_images
from data_integrity_check import get_media_sizes
from select_resize_deploy import (
    create_flir_adas_final_data_set,
    create_final_falling_human_data_set,
    create_PST900_final_data_set,
)
from final_data_set_creater import (
    create_final_training_set,
    create_final_validation_set,
)
from yolo_annotation_path import create_yolo_dataset_structure


def main(parent_dir):
    """parent path is the thermal_rescue_robot main directory,the one that contains the setup.py file"""
    # download the main datasets:falling_humans, PST900 and flir_ADAS
    datasets_path = os.path.join(parent_dir, "datasets")
    os.makedirs(datasets_path, exist_ok=True)
    print(f"Datasets will be downloaded to: {datasets_path}")
    print("Starting falling human dataset download...")
    download_falling_human(os.path.join(datasets_path, "archive"))
    print("Starting PST900 dataset download...")
    download_PST900(os.path.join(datasets_path, "real_data"))
    print("Starting FLIR ADAS dataset download...")
    download_flir_adas_dataset(datasets_path)
    print("Training datasets downloaded successfully!")
    # downloading the test dataset
    print("Starting test dataset download...")
    download_test_images(datasets_path)
    print(f"Test dataset downloaded successfully! in {datasets_path}")
    # now we do the data preprocessing
    print("Starting data preprocessing...")
    data_paths = []
    # let's add flir adas images
    data_paths.append(
        os.path.join(datasets_path, "archive/FLIR_ADAS_v2/images_thermal_train/data")
    )
    data_paths.append(
        os.path.join(datasets_path, "archive/FLIR_ADAS_v2/images_thermal_val/data")
    )
    # let's add PST900 images
    data_paths.append(
        os.path.join(datasets_path, "real_data/PST900_RGBT_Dataset/train/thermal")
    )
    data_paths.append(
        os.path.join(datasets_path, "real_data/PST900_RGBT_Dataset/test/thermal")
    )
    # let's add falling human images
    data_paths.append(os.path.join(datasets_path, "falling humans/train"))
    data_paths.append(os.path.join(datasets_path, "falling humans/test"))
    # defining the norm for each dataset
    dataset_norms = [
        (640, 512),
        (640, 480),
        (1280, 720),
    ]  # (flir adas, falling human, PST900)
    for i in range(0, len(data_paths), 2):
        size_im1, size_vid1 = get_media_sizes(data_paths[i])
        size_im2, size_vid2 = get_media_sizes(data_paths[i + 1])
        # we will check the images sizes, if they are not the same we will remove the outliers
        s1 = False
        s2 = False
        for size in size_im1:
            if size not in dataset_norms:
                s1 = True
                break
        for size in size_im2:
            if size not in dataset_norms:
                s2 = True
                break
        # if there are outliers we will remove them
        if s1:
            print(f"Outliers found in {data_paths[i]}. Removing them...")
            for image in os.listdir(data_paths[i]):
                img_path = os.path.join(data_paths[i], image)
                with Image.open(img_path) as img:
                    size = img.size
                if size not in dataset_norms:
                    os.remove(img_path)
            print(f"Outliers removed from {data_paths[i]}.")
        if s2:
            print(f"Outliers found in {data_paths[i+1]}. Removing them...")
            for image in os.listdir(data_paths[i + 1]):
                img_path = os.path.join(data_paths[i + 1], image)
                with Image.open(img_path) as img:
                    size = img.size
                if size not in dataset_norms:
                    os.remove(img_path)
            print(f"Outliers removed from {data_paths[i+1]}.")
    print("Data preprocessing completed successfully!")
    # now we create the final dataset with unified image resolutions
    # select_resize_deploy do the unification on the fly and only on the selected images,
    # which is more efficient than unifying all the images and then selecting the ones we need
    print(
        "Starting creating the final flir adas dataset with unified image resolutions..."
    )
    if not os.path.exists(os.path.join(datasets_path, "final_datasets/flir_adas")):
        os.makedirs(os.path.join(datasets_path, "final_datasets/flir_adas/train"))
        os.makedirs(os.path.join(datasets_path, "final_datasets/flir_adas/val"))
    create_flir_adas_final_data_set(
        300,
        50,
        os.path.join(datasets_path, "archive/FLIR_ADAS_v2/images_thermal_train"),
        os.path.join(datasets_path, "archive/FLIR_ADAS_v2/images_thermal_val"),
        os.path.join(datasets_path, "final_datasets/flir_adas/train"),
        os.path.join(datasets_path, "final_datasets/flir_adas/val"),
    )
    print("Final flir adas dataset created successfully!")
    print(
        "Starting creating the final falling human dataset with unified image resolutions..."
    )
    if not os.path.exists(os.path.join(datasets_path, "final_datasets/falling_human")):
        os.makedirs(
            os.path.join(datasets_path, "final_datasets/falling_human/training")
        )
        os.makedirs(os.path.join(datasets_path, "final_datasets/falling_human/testing"))
    create_final_falling_human_data_set(
        os.path.join(datasets_path, "falling humans"),
        os.path.join(datasets_path, "final_datasets/falling_human/training"),
        os.path.join(datasets_path, "final_datasets/falling_human/testing"),
        300,
        50,
    )
    print("Final falling human dataset created successfully!")
    print(
        "Starting creating the final PST900 dataset with unified image resolutions..."
    )
    if not os.path.exists(os.path.join(datasets_path, "final_datasets/PST900")):
        os.makedirs(os.path.join(datasets_path, "final_datasets/PST900/training"))
        os.makedirs(os.path.join(datasets_path, "final_datasets/PST900/testing"))
    create_PST900_final_data_set(
        os.path.join(datasets_path, "final_datasets/PST900"),
        os.path.join(datasets_path, "real_data/PST900_RGBT_Dataset/train/thermal"),
        os.path.join(datasets_path, "real_data/PST900_RGBT_Dataset/train/labels"),
        os.path.join(datasets_path, "real_data/PST900_RGBT_Dataset/test/thermal"),
        os.path.join(datasets_path, "real_data/PST900_RGBT_Dataset/test/labels"),
    )
    print("Final PST900 dataset created successfully!")
    # now we have the final datasets , time to merge them into one dataset:the final_data_set
    print("Starting merging the final datasets into one dataset...")
    if not os.path.exists(os.path.join(datasets_path, "final_datasets/final_data_set")):
        os.makedirs(
            os.path.join(datasets_path, "final_datasets/final_data_set/training")
        )
        os.makedirs(
            os.path.join(datasets_path, "final_datasets/final_data_set/validation")
        )
    create_final_training_set(
        os.path.join(datasets_path, "final_datasets/final_data_set"),
        os.path.join(datasets_path, "final_datasets/flir_adas"),
        os.path.join(datasets_path, "final_datasets/falling_human"),
        os.path.join(datasets_path, "final_datasets/PST900"),
    )
    create_final_validation_set(
        os.path.join(datasets_path, "final_datasets/final_data_set"),
        os.path.join(datasets_path, "final_datasets/flir_adas"),
        os.path.join(datasets_path, "final_datasets/falling_human"),
        os.path.join(datasets_path, "final_datasets/PST900"),
    )
    print("Final dataset created successfully!")
    # now we create the yolo specific dataset
    yolo_path = os.path.join(datasets_path, "yolo_dataset")
    if not os.path.exists(yolo_path):
        os.makedirs(yolo_path)
    create_yolo_dataset_structure(
        yolo_path,
        os.path.join(datasets_path, "final_datasets"),
        os.path.join(datasets_path, "test_images"),
    )
    print(f"Yolo dataset structure created successfully in {yolo_path}!")
