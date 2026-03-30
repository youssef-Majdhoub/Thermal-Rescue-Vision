import os
import shutil
import yaml
from tqdm import tqdm


def create_yolo_dataset_structure(main_folder, test_path=None):
    """this function will create the dataset structure for the yolo format.
    it will create a main folder with two subfolders one for training and one for validation.
    each subfolder will contain two subfolders one for images and one for labels.
    it will also create a yaml file that contains the dataset information.
    after creating the dataset structure we will copy the images and labels from the
    final_data_set to the new dataset structure"""
    training_folder = os.path.join(main_folder, "training")
    os.makedirs(training_folder, exist_ok=True)
    validation_folder = os.path.join(main_folder, "validation")
    os.makedirs(validation_folder, exist_ok=True)
    training_images_path = os.path.join(training_folder, "images")
    training_labels_path = os.path.join(training_folder, "labels")
    validation_images_path = os.path.join(validation_folder, "images")
    validation_labels_path = os.path.join(validation_folder, "labels")
    os.makedirs(training_images_path, exist_ok=True)
    os.makedirs(training_labels_path, exist_ok=True)
    os.makedirs(validation_images_path, exist_ok=True)
    os.makedirs(validation_labels_path, exist_ok=True)
    # creating .yaml file
    yaml_content = {
        "path": main_folder,
        "train": "training/images",
        "val": "validation/images",
        "test": "test/images",
        "nc": 1,  # number of classes
        "names": ["human"],  # class names
    }
    yaml_file_path = os.path.join(main_folder, "dataset.yaml")
    with open(yaml_file_path, "w") as yaml_file:
        yaml.dump(yaml_content, yaml_file, sort_keys=False)
    print(f"Dataset structure created at: {main_folder}")
    # the sources paths
    training_source_path = r".\final_data_sets\final_data_set\training"
    validation_source_path = r".\final_data_sets\final_data_set\validation"
    training_images_source = os.path.join(training_source_path, "data")
    training_labels_source = os.path.join(
        training_source_path, "yolo_annotations", "data"
    )
    validation_images_source = os.path.join(validation_source_path, "data")
    validation_labels_source = os.path.join(
        validation_source_path, "yolo_annotations", "data"
    )
    # copying training images and labels
    for filename in tqdm(
        os.listdir(training_images_source), desc="Copying training data"
    ):
        label_name = os.path.splitext(filename)[0] + ".txt"
        if os.path.exists(os.path.join(training_labels_source, label_name)):
            shutil.copy(
                os.path.join(training_images_source, filename),
                os.path.join(training_images_path, filename),
            )
            shutil.copy(
                os.path.join(training_labels_source, label_name),
                os.path.join(training_labels_path, label_name),
            )
        else:
            print(f"Label file {label_name} not found for image {filename}")
    # copying validation images and labels
    for filename in tqdm(
        os.listdir(validation_images_source), desc="Copying validation data"
    ):
        label_name = os.path.splitext(filename)[0] + ".txt"
        if os.path.exists(os.path.join(validation_labels_source, label_name)):
            shutil.copy(
                os.path.join(validation_images_source, filename),
                os.path.join(validation_images_path, filename),
            )
            shutil.copy(
                os.path.join(validation_labels_source, label_name),
                os.path.join(validation_labels_path, label_name),
            )
        else:
            print(f"Label file {label_name} not found for image {filename}")
    # coping test images and labels from falling human dataset
    if test_path is None:
        test_source_path = os.path.abspath(
            "./falling humans"
        )  # this is jsut images for deplyment testing
    else:
        test_source_path = test_path
    shutil.copytree(
        test_source_path, os.path.join(main_folder, "test"), dirs_exist_ok=True
    )


if __name__ == "__main__":
    main_folder = "C:\\Users\\medbe\\OneDrive\\Bureau\\PFA2026\\yolo_dataset"
    os.makedirs(main_folder, exist_ok=True)
    create_yolo_dataset_structure(main_folder)
