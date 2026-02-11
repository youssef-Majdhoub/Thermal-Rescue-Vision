from data_selection_from_FLIR_ADAS import select_images_from_flir_adas
from annotation_conversion_script import convert_coco_to_yolo
import os


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
        output_dir=yolo_tranning_output_dir,
        images_path=new_tranning_dir,
        needed_categories=[0],
    )
    convert_coco_to_yolo(
        coco_annotation_path=os.path.join(new_test_dir, "coco.json"),
        output_dir=yolo_testing_output_dir,
        images_path=new_test_dir,
        needed_categories=[0],
    )
