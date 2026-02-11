import json
import os
import shutil
import random
from tqdm import tqdm


def select_images_from_flir_adas(
    image_source_dir,
    coco_annotation_path,
    output_dir,
    needed_categories_id,
    new_categories_id_map,  # e.g., {1: 0, 73: 0, 74: 0}
    num_images=2000,
    bias=0.8,
):
    """I want to create a dataset with 80% images containing humans and 20% without.
    This function will read the original COCO JSON, classify images,
    and then create a new dataset accordingly.
    Args:
        image_source_dir (str): Directory where original images are stored.
        coco_annotation_path (str): Path to the original COCO JSON file.
        output_dir (str): Directory where the new dataset will be saved.
        needed_categories_id (list): List of category IDs that count as needed (e.g., humans).
        new_categories_id_map (dict): Mapping from old category IDs to new ones (e.g., {1: 0, 73: 0, 74: 0}).
        num_images (int): Total number of images to select.
        bias (float): Proportion of images containing needed categories (e.g., 0.8 for 80%).
    """

    # --- 1. Setup Output Directories ---
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    dest_images_dir = os.path.join(output_dir, "images")
    dest_json_path = os.path.join(output_dir, "coco.json")

    if not os.path.exists(dest_images_dir):
        os.makedirs(dest_images_dir)

    print(f"Loading COCO JSON from: {coco_annotation_path}")
    with open(coco_annotation_path, "r") as f:
        coco_data = json.load(f)

    # --- 2. Optimization: Index Annotations by Image ID ---
    # This turns a slow loop O(N^2) into a fast one O(N)
    print("Indexing annotations...")
    img_id_to_anns = {}
    for ann in coco_data["annotations"]:
        img_id = ann["image_id"]
        if img_id not in img_id_to_anns:
            img_id_to_anns[img_id] = []
        img_id_to_anns[img_id].append(ann)

    # --- 3. Classify Images: Target vs. Noise ---
    target_images = []  # Contains humans
    noise_images = []  # Empty or contains only other objects (cars, dogs, etc)

    print("Categorizing images...")
    for img in coco_data["images"]:
        img_id = img["id"]
        anns = img_id_to_anns.get(img_id, [])

        # Check if this image has at least one "Human" (needed category)
        has_needed_category = False
        for ann in anns:
            if ann["category_id"] in needed_categories_id:
                has_needed_category = True
                break

        if has_needed_category:
            target_images.append(img)
        else:
            noise_images.append(img)

    # --- 4. Select the Dataset (Math) ---
    target_count = int(num_images * bias)
    noise_count = num_images - target_count

    # Handle case where we don't have enough images
    if len(target_images) < target_count:
        print(
            f"Warning: Requested {target_count} targets but only found {len(target_images)}. Taking all."
        )
        target_count = len(target_images)

    if len(noise_images) < noise_count:
        print(
            f"Warning: Requested {noise_count} noise images but only found {len(noise_images)}. Taking all."
        )
        noise_count = len(noise_images)

    # Randomly sample
    random.seed(42)
    selected_targets = random.sample(target_images, target_count)
    selected_noise = random.sample(noise_images, noise_count)

    final_image_list = selected_targets + selected_noise
    print(
        f"Final Selection: {len(selected_targets)} Targets + {len(selected_noise)} Noise = {len(final_image_list)} Total"
    )

    # --- 5. Build New Annotations (Crucial Step) ---
    final_annotations = []
    new_ann_id = 1

    print("Processing annotations...")
    for img in final_image_list:
        # If it's a NOISE image, we skip this loop entirely (Result: 0 annotations)
        if img in selected_noise:
            continue

        # If it's a TARGET image, we only keep the needed categories and remap IDs
        original_anns = img_id_to_anns.get(img["id"], [])

        for ann in original_anns:
            old_cat_id = ann["category_id"]

            if old_cat_id in needed_categories_id:
                # Create a fresh copy of the annotation
                new_ann = ann.copy()

                # Update ID to the new mapped ID (e.g., 1 -> 0)
                new_ann["category_id"] = new_categories_id_map[old_cat_id]

                # Give it a clean, sequential ID
                new_ann["id"] = new_ann_id

                final_annotations.append(new_ann)
                new_ann_id += 1

    # Define the single category for the new JSON
    new_categories = [{"id": 0, "name": "person", "supercategory": "none"}]

    new_coco = {
        "info": coco_data.get("info", {}),
        "licenses": coco_data.get("licenses", []),
        "images": final_image_list,
        "annotations": final_annotations,
        "categories": new_categories,
    }

    # --- 6. Save JSON & Copy Files ---
    print(f"Saving new COCO JSON to {dest_json_path}...")
    with open(dest_json_path, "w") as f:
        json.dump(new_coco, f, indent=4)

    print(f"Copying {len(final_image_list)} images to {dest_images_dir}...")
    for img in tqdm(final_image_list, desc="Copying"):
        src_path = os.path.join(image_source_dir, img["file_name"])
        dst_path = os.path.join(dest_images_dir, img["file_name"])

        # Handle subfolders if necessary (e.g. if filename is "data/img1.jpg")
        os.makedirs(os.path.dirname(dst_path), exist_ok=True)

        if os.path.exists(src_path):
            shutil.copy(src_path, dst_path)
        else:
            print(f"Missing file: {src_path}")

    print("Done! Dataset created successfully.")


if __name__ == "__main__":
    # --- CONFIGURATION ---
    # Use raw strings (r"") for Windows paths
    img_path = r"C:\Users\medbe\OneDrive\Bureau\PFA2026\archive\FLIR_ADAS_v2\images_thermal_train\data"
    json_path = r"C:\Users\medbe\OneDrive\Bureau\PFA2026\archive\FLIR_ADAS_v2\images_thermal_train\coco.json"
    out_dir = r"C:\Users\medbe\OneDrive\Bureau\PFA2026\final_data_sets\Flir_traning"

    # Define which IDs are humans in your source dataset
    human_ids = [1, 73, 74]

    # Define how to map them (All become class 0)
    mapping = {1: 0, 73: 0, 74: 0}

    select_images_from_flir_adas(
        image_source_dir=img_path,
        coco_annotation_path=json_path,
        output_dir=out_dir,
        needed_categories_id=human_ids,
        new_categories_id_map=mapping,
        num_images=2000,
        bias=0.8,
    )
