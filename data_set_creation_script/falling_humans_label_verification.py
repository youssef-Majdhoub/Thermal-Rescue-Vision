import os
import shutil


def verify_labels_and_images(falling_humans_folder=None):
    if falling_humans_folder is None:
        falling_humans_folder = os.path.abspath("./falling humans")
    labels_folders = []
    labels_folders.append(os.path.join(falling_humans_folder, "labels"))
    labels_folders.append(
        os.path.join(falling_humans_folder, "train", "label", "obj_Train_data")
    )
    images_folders = []
    images_folders.append(os.path.join(falling_humans_folder, "images"))
    images_folders.append(os.path.join(falling_humans_folder, "train", "image"))
    images_folders.append(
        os.path.join(falling_humans_folder, "validation", "image val")
    )
    # check for existence of labels and images folders
    for folder in labels_folders:
        if not os.path.isdir(folder):
            print(f"Error: The provided path {folder} is not a valid directory.")
        else:
            print(f"Labels folder found: {folder}")
    for folder in images_folders:
        if not os.path.isdir(folder):
            print(f"Error: The provided path {folder} is not a valid directory.")
        else:
            print(f"Images folder found: {folder}")
    # cheking if each image has a corresponding label file
    missing_labels_count = 0
    duplicates_count = 0
    empty_labels_count = 0
    no_label_files_path = os.path.join(falling_humans_folder, "no_label_files")
    final_images_folder = os.path.join(falling_humans_folder, "final_images")
    final_labels_folder = os.path.join(falling_humans_folder, "final_labels")
    os.makedirs(no_label_files_path, exist_ok=True)
    os.makedirs(final_images_folder, exist_ok=True)
    os.makedirs(final_labels_folder, exist_ok=True)
    for images_folder in images_folders:
        for image_file in os.listdir(images_folder):
            if image_file.endswith((".jpg", ".jpeg", ".png")):
                label_file = os.path.splitext(image_file)[0] + ".txt"
                for label_folder in labels_folders:
                    label_file_path = os.path.join(label_folder, label_file)
                    if os.path.isfile(label_file_path):
                        # If label file exists, copy both image and label to final folders
                        if os.path.isfile(
                            os.path.join(final_images_folder, image_file)
                        ):
                            print(
                                f"Duplicate label file found: {label_file}. Skipping {image_file}."
                            )
                            duplicates_count += 1
                            break
                        with open(label_file_path, "r") as f:
                            if f.read().strip() == "":
                                print(
                                    f"Empty label file found: {label_file}. Skipping {image_file}."
                                )
                                empty_labels_count += 1
                                break
                        shutil.copy(
                            os.path.join(images_folder, image_file),
                            os.path.join(final_images_folder, image_file),
                        )
                        shutil.copy(
                            label_file_path,
                            os.path.join(final_labels_folder, label_file),
                        )
                        break
        else:
            # If no label file is found, move the image to the no_label_files folder
            shutil.move(
                os.path.join(images_folder, image_file),
                os.path.join(no_label_files_path, image_file),
            )
            print(
                f"No label file found for {image_file}. Moved to {no_label_files_path}."
            )
            missing_labels_count += 1
    print(
        "Label verification completed. Check the final_images and final_labels folders for verified data."
    )
    print(f"Total images without labels: {missing_labels_count}")
    print(f"Total duplicate images files: {duplicates_count}")


if __name__ == "__main__":
    falling_humans_folder = r"C:\Users\medbe\OneDrive\Bureau\PFA2026\falling humans"
    verify_labels_and_images(falling_humans_folder)
