from data_selection_from_FLIR_ADAS import select_images_from_flir_adas
from annotation_conversion_script import convert_coco_to_yolo, convert_yolo_to_coco
from image_resolution_unifier import unify_image_resolutions
from yolo_annotation_from_label_images import create_yolo_annotations_from_label_images
import os
import shutil


def create_flir_adas_final_data_set(new_traning_size=2000, new_test_size=200):
    """those images are already 640x512,
    so we can just copy them to the output directory without resizing or padding.
    we select a subset of thsoe images so we done need the whole dataset for training nor testing.
    """
    old_tranning_dir = "C:/Users/medbe/OneDrive/Bureau/PFA2026/archive/FLIR_ADAS_v2/images_thermal_train"
    old_test_dir = (
        "C:/Users/medbe/OneDrive/Bureau/PFA2026/archive/FLIR_ADAS_v2/images_thermal_val"
    )
    new_tranning_dir = (
        "C:/Users/medbe/OneDrive/Bureau/PFA2026/final_data_sets/Flir_data_set/training"
    )
    new_test_dir = (
        "C:/Users/medbe/OneDrive/Bureau/PFA2026/final_data_sets/Flir_data_set/testing"
    )
    old_tranning_images_path = os.path.join(old_tranning_dir, "data")
    old_tranning_json_path = os.path.join(old_tranning_dir, "coco.json")
    old_test_images_path = os.path.join(old_test_dir, "data")
    old_test_json_path = os.path.join(old_test_dir, "coco.json")
    human_ids = [1, 73, 74]
    mapping = {1: 0, 73: 0, 74: 0}
    select_images_from_flir_adas(
        image_source_dir=old_tranning_images_path,
        coco_annotation_path=old_tranning_json_path,
        output_dir=new_tranning_dir,
        needed_categories_id=human_ids,
        new_categories_id_map=mapping,
        num_images=new_traning_size,
    )
    select_images_from_flir_adas(
        image_source_dir=old_test_images_path,
        coco_annotation_path=old_test_json_path,
        output_dir=new_test_dir,
        needed_categories_id=human_ids,
        new_categories_id_map=mapping,
        num_images=new_test_size,
    )
    yolo_tranning_output_dir = os.path.join(new_tranning_dir, "yolo_format")
    yolo_testing_output_dir = os.path.join(new_test_dir, "yolo_format")

    convert_coco_to_yolo(
        coco_annotation_path=os.path.join(new_tranning_dir, "coco.json"),
        output_annotation_path=yolo_tranning_output_dir,
        images_path=new_tranning_dir,
        needed_categories=[0],
    )
    convert_coco_to_yolo(
        coco_annotation_path=os.path.join(new_test_dir, "coco.json"),
        output_annotation_path=yolo_testing_output_dir,
        images_path=new_test_dir,
        needed_categories=[0],
    )


def create_PST900_final_data_set():
    """those images are 1280x720, so we need to resize them to 640x512 with padding.
    there is no coco nor yolo annotations for this dataset, we need to resize then then create the coco and yolo annotations from scratch.
    for that purpose we will use temprary directories to store the resized images and the label images which are also resized to 640x512
    then we will create the coco and yolo annotations from those resized images and label images.
    then we will delete the temprary resized_PST900 directory and create the PST900_data_set.
    all temprary and final directories will contain two subdirectories one for training and one for testing.
    """
    main_path = "C:/Users/medbe/OneDrive/Bureau/PFA2026/final_data_sets"
    temprary_resized_dir = os.path.join(main_path, "temprary_resized_PST900")
    final_pst900_dir = os.path.join(main_path, "PST900_data_set")
    if not os.path.exists(temprary_resized_dir):
        os.makedirs(temprary_resized_dir)
    if not os.path.exists(final_pst900_dir):
        os.makedirs(final_pst900_dir)
    training_temprary_dir = os.path.join(temprary_resized_dir, "training")
    testing_temprary_dir = os.path.join(temprary_resized_dir, "testing")
    training_final_dir = os.path.join(final_pst900_dir, "training")
    testing_final_dir = os.path.join(final_pst900_dir, "testing")
    if not os.path.exists(training_temprary_dir):
        os.makedirs(training_temprary_dir)
    if not os.path.exists(testing_temprary_dir):
        os.makedirs(testing_temprary_dir)
    if not os.path.exists(training_final_dir):
        os.makedirs(training_final_dir)
    if not os.path.exists(testing_final_dir):
        os.makedirs(testing_final_dir)
    # here we will resize the images and the label images to 640x512 with padding
    orginal_training_images_dir = "C:\\Users\\medbe\\OneDrive\\Bureau\\PFA2026\\real_data\\PST900_RGBT_Dataset\\train\\thermal"
    orginal_training_labels_dir = "C:\\Users\\medbe\\OneDrive\\Bureau\\PFA2026\\real_data\\PST900_RGBT_Dataset\\train\\labels"
    orginal_testing_images_dir = "C:\\Users\\medbe\\OneDrive\\Bureau\\PFA2026\\real_data\\PST900_RGBT_Dataset\\test\\thermal"
    orginal_testing_labels_dir = "C:\\Users\\medbe\\OneDrive\\Bureau\\PFA2026\\real_data\\PST900_RGBT_Dataset\\test\\labels"
    resized_training_images_dir = os.path.join(training_temprary_dir, "data")
    resized_training_labels_dir = os.path.join(training_temprary_dir, "labels")
    resized_testing_images_dir = os.path.join(testing_temprary_dir, "data")
    resized_testing_labels_dir = os.path.join(testing_temprary_dir, "labels")
    if not os.path.exists(resized_training_images_dir):
        os.makedirs(resized_training_images_dir)
    if not os.path.exists(resized_training_labels_dir):
        os.makedirs(resized_training_labels_dir)
    if not os.path.exists(resized_testing_images_dir):
        os.makedirs(resized_testing_images_dir)
    if not os.path.exists(resized_testing_labels_dir):
        os.makedirs(resized_testing_labels_dir)
    # resize training images and labels
    unify_image_resolutions(
        input_dir=orginal_training_images_dir,
        output_dir=resized_training_images_dir,
    )
    unify_image_resolutions(
        input_dir=orginal_training_labels_dir,
        output_dir=resized_training_labels_dir,
    )
    # resize testing images and labels
    unify_image_resolutions(
        input_dir=orginal_testing_images_dir,
        output_dir=resized_testing_images_dir,
    )
    unify_image_resolutions(
        input_dir=orginal_testing_labels_dir,
        output_dir=resized_testing_labels_dir,
    )
    # create yolo annotations from the resized label images
    training_yolo_annotation_path = os.path.join(training_final_dir, "yolo_annotations")
    if not os.path.exists(training_yolo_annotation_path):
        os.makedirs(training_yolo_annotation_path)
    create_yolo_annotations_from_label_images(
        label_images_dir=resized_training_labels_dir,
        output_annotation_dir=training_yolo_annotation_path,
        needed_indexes=[4],
    )
    testing_yolo_annotation_path = os.path.join(testing_final_dir, "yolo_annotations")
    if not os.path.exists(testing_yolo_annotation_path):
        os.makedirs(testing_yolo_annotation_path)
    create_yolo_annotations_from_label_images(
        label_images_dir=resized_testing_labels_dir,
        output_annotation_dir=testing_yolo_annotation_path,
        needed_indexes=[4],
    )
    # copy the resized images to the final directory
    final_traning_data_path = os.path.join(training_final_dir, "data")
    final_testing_data_path = os.path.join(testing_final_dir, "data")
    if not os.path.exists(final_traning_data_path):
        os.makedirs(final_traning_data_path)
    if not os.path.exists(final_testing_data_path):
        os.makedirs(final_testing_data_path)
    for file_name in os.listdir(resized_training_images_dir):
        shutil.copy(
            os.path.join(resized_training_images_dir, file_name),
            os.path.join(final_traning_data_path, file_name),
        )
    for file_name in os.listdir(resized_testing_images_dir):
        shutil.copy(
            os.path.join(resized_testing_images_dir, file_name),
            os.path.join(final_testing_data_path, file_name),
        )
    # we can delete the temprary resized directory after we have created the final dataset
    shutil.rmtree(temprary_resized_dir)
    # we will also create coco annotations from the yolo annotations
    convert_yolo_to_coco(
        yolo_annotation_path=training_yolo_annotation_path,
        images_path=final_traning_data_path,
        output_json_path=os.path.join(training_final_dir, "coco.json"),
    )
    convert_yolo_to_coco(
        yolo_annotation_path=testing_yolo_annotation_path,
        images_path=final_testing_data_path,
        output_json_path=os.path.join(testing_final_dir, "coco.json"),
    )


if __name__ == "__main__":
    # create_flir_adas_final_data_set()
    # create_PST900_final_data_set()
    pass
