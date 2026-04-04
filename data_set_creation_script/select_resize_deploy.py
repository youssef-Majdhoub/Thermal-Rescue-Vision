from data_selection_from_FLIR_ADAS import select_images_from_flir_adas
from annotation_conversion_script import convert_coco_to_yolo, convert_yolo_to_coco
from image_resolution_unifier import unify_image_resolutions
from yolo_annotation_from_label_images import create_yolo_annotations_from_label_images
import os
import shutil
from tqdm import tqdm


def create_flir_adas_final_data_set(
    new_traning_size=300,
    new_test_size=50,
    old_tranning_dir=None,
    old_test_dir=None,
    new_tranning_dir=None,
    new_test_dir=None,
):
    """those images are already 640x512,
    so we can just copy them to the output directory without resizing or padding.
    we select a subset of thsoe images so we done need the whole dataset for training nor testing.
    """
    if old_tranning_dir is None:
        old_tranning_dir = "C:/Users/medbe/OneDrive/Bureau/PFA2026/archive/FLIR_ADAS_v2/images_thermal_train"
    if old_test_dir is None:
        old_test_dir = "C:/Users/medbe/OneDrive/Bureau/PFA2026/archive/FLIR_ADAS_v2/images_thermal_val"
    if new_tranning_dir is None:
        new_tranning_dir = "C:/Users/medbe/OneDrive/Bureau/PFA2026/final_data_sets/Flir_data_set/training"
    if new_test_dir is None:
        new_test_dir = "C:/Users/medbe/OneDrive/Bureau/PFA2026/final_data_sets/Flir_data_set/testing"
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
    yolo_tranning_output_dir = os.path.join(new_tranning_dir, "yolo_annotations")
    yolo_testing_output_dir = os.path.join(new_test_dir, "yolo_annotations")

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


def create_PST900_final_data_set(
    main_path=None,
    original_training_images_dir=None,
    original_training_labels_dir=None,
    original_testing_images_dir=None,
    original_testing_labels_dir=None,
):
    """those images are 1280x720, so we need to resize them to 640x512 with padding.
    there is no coco nor yolo annotations for this dataset, we need to resize then then create the coco and yolo annotations from scratch.
    for that purpose we will use temprary directories to store the resized images and the label images which are also resized to 640x512
    then we will create the coco and yolo annotations from those resized images and label images.
    then we will delete the temprary resized_PST900 directory and create the PST900_data_set.
    all temprary and final directories will contain two subdirectories one for training and one for testing.
    """
    if main_path is None:
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
    if original_training_images_dir is None:
        original_training_images_dir = "C:\\Users\\medbe\\OneDrive\\Bureau\\PFA2026\\real_data\\PST900_RGBT_Dataset\\train\\thermal"
    if original_training_labels_dir is None:
        original_training_labels_dir = "C:\\Users\\medbe\\OneDrive\\Bureau\\PFA2026\\real_data\\PST900_RGBT_Dataset\\train\\labels"
    if original_testing_images_dir is None:
        original_testing_images_dir = "C:\\Users\\medbe\\OneDrive\\Bureau\\PFA2026\\real_data\\PST900_RGBT_Dataset\\test\\thermal"
    if original_testing_labels_dir is None:
        original_testing_labels_dir = "C:\\Users\\medbe\\OneDrive\\Bureau\\PFA2026\\real_data\\PST900_RGBT_Dataset\\test\\labels"
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
        input_dir=original_training_images_dir,
        output_dir=resized_training_images_dir,
    )
    unify_image_resolutions(
        input_dir=original_training_labels_dir,
        output_dir=resized_training_labels_dir,
    )
    # resize testing images and labels
    unify_image_resolutions(
        input_dir=original_testing_images_dir,
        output_dir=resized_testing_images_dir,
    )
    unify_image_resolutions(
        input_dir=original_testing_labels_dir,
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


def create_final_falling_human_data_set(
    original_dir=None,
    final_training_dir=None,
    final_testing_dir=None,
    num_training_images=300,
    num_testing_images=100,
):
    """this function will create the final falling human dataset , the yolo annotations for this dataset
    are already created but stored badly in one directory for both training and testing images, so we will
    need to separate the annotations for training and testing images and store them in the final dataset
    also we will need to make sure the category is stored in obj.names.txt file in the final dataset directory
    and we need to create coco annotations from the yolo annotations and store them"""
    # path to the original dataset
    if original_dir is None:
        original_dir = "C:\\Users\\medbe\\OneDrive\\Bureau\\PFA2026\\falling humans"
    original_training_dir = os.path.join(original_dir, "train")
    original_testing_dir = os.path.join(original_dir, "validation")
    original_yolo_annotations_dir = os.path.join(
        original_training_dir, "label", "obj_Train_data"
    )
    original_training_images_dir = os.path.join(original_training_dir, "image")
    original_testing_images_dir = os.path.join(original_testing_dir, "image val")
    # create the final dataset directories
    if final_training_dir is None:
        final_training_dir = os.path.join(
            "C:\\Users\\medbe\\OneDrive\\Bureau\\PFA2026\\final_data_sets",
            "falling_human",
            "training",
        )
    if final_testing_dir is None:
        final_testing_dir = os.path.join(
            "C:\\Users\\medbe\\OneDrive\\Bureau\\PFA2026\\final_data_sets",
            "falling_human",
            "testing",
        )
    if not os.path.exists(final_training_dir):
        os.makedirs(final_training_dir)
    if not os.path.exists(final_testing_dir):
        os.makedirs(final_testing_dir)
    # copy the training images and annotations to the final dataset directory
    final_training_images_dir = os.path.join(final_training_dir, "data")
    final_training_annotations_dir = os.path.join(
        final_training_dir, "yolo_annotations"
    )
    final_training_yolo_data_path = os.path.join(final_training_annotations_dir, "data")
    final_testing_annotations_dir = os.path.join(final_testing_dir, "yolo_annotations")
    final_testing_yolo_data_path = os.path.join(final_testing_annotations_dir, "data")
    if not os.path.exists(final_training_images_dir):
        os.makedirs(final_training_images_dir)
    if not os.path.exists(final_training_annotations_dir):
        os.makedirs(final_training_annotations_dir)
    if not os.path.exists(final_training_yolo_data_path):
        os.makedirs(final_training_yolo_data_path)
    if not os.path.exists(final_testing_yolo_data_path):
        os.makedirs(final_testing_yolo_data_path)
    s = 0
    for file_name in tqdm(
        os.listdir(original_training_images_dir),
        desc="Copying training images and annotations",
    ):
        if s >= num_training_images:
            break
        s += 1
        annotation_file_name = os.path.splitext(file_name)[0] + ".txt"
        shutil.copy(
            os.path.join(original_training_images_dir, file_name),
            os.path.join(final_training_images_dir, file_name),
        )
        if not os.path.exists(
            os.path.join(original_yolo_annotations_dir, annotation_file_name)
        ):
            print(
                f"Annotation file {annotation_file_name} not found for image {file_name}"
            )
            with open(
                os.path.join(final_training_yolo_data_path, annotation_file_name), "w"
            ) as f:
                pass  # create an empty annotation file
            continue
        shutil.copy(
            os.path.join(original_yolo_annotations_dir, annotation_file_name),
            os.path.join(final_training_yolo_data_path, annotation_file_name),
        )
    # copy the testing images to the final dataset directory
    final_testing_images_dir = os.path.join(final_testing_dir, "data")
    final_testing_annotations_dir = os.path.join(final_testing_dir, "yolo_annotations")
    if not os.path.exists(final_testing_images_dir):
        os.makedirs(final_testing_images_dir)
    if not os.path.exists(final_testing_annotations_dir):
        os.makedirs(final_testing_annotations_dir)
    s = 0
    for file_name in tqdm(
        os.listdir(original_testing_images_dir),
        desc="Copying testing images and annotations",
    ):
        if s >= num_testing_images:
            break
        s += 1
        shutil.copy(
            os.path.join(original_testing_images_dir, file_name),
            os.path.join(final_testing_images_dir, file_name),
        )
        annotation_file_name = os.path.splitext(file_name)[0] + ".txt"
        if not os.path.exists(
            os.path.join(original_yolo_annotations_dir, annotation_file_name)
        ):
            print(
                f"Annotation file {annotation_file_name} not found for image {file_name}"
            )
            with open(
                os.path.join(final_testing_yolo_data_path, annotation_file_name), "w"
            ) as f:
                pass  # create an empty annotation file
            continue
        shutil.copy(
            os.path.join(original_yolo_annotations_dir, annotation_file_name),
            os.path.join(final_testing_yolo_data_path, annotation_file_name),
        )
    # create obj_names.txt file in the final dataset directories
    obj_names_file = os.path.join(final_training_annotations_dir, "obj.names")
    with open(obj_names_file, "w") as f:
        f.write("human\n")
    obj_names_file = os.path.join(final_testing_annotations_dir, "obj.names")
    with open(obj_names_file, "w") as f:
        f.write("human\n")
    # create coco annotations
    convert_yolo_to_coco(
        yolo_annotation_path=final_training_annotations_dir,
        images_path=final_training_images_dir,
        output_json_path=os.path.join(final_training_dir, "coco.json"),
    )
    convert_yolo_to_coco(
        yolo_annotation_path=final_testing_annotations_dir,
        images_path=final_testing_images_dir,
        output_json_path=os.path.join(final_testing_dir, "coco.json"),
    )


if __name__ == "__main__":
    create_flir_adas_final_data_set()
    create_PST900_final_data_set()
    create_final_falling_human_data_set()
