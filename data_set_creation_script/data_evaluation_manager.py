import os
import json
import pandas as pd, numpy as np

# --- STRICT "LIVING THINGS ONLY" MAPPING ---
# Class 0 = Humans
# Class 1 = living creatues

ID_MAPPING = {
    # === HUMANS (Class 0) ===
    1: 0,  # Person (Pedestrian)
    73: 0,  # Stroller (Contains a human baby)
    74: 0,  # Rider (The human ON the bike/motorcycle)
    # === LIVING CREATURES (Class 1) ===
    15: 1,  # Bird (Optional: delete this line if you don't want birds)
    16: 1,  # Cat
    17: 1,  # Dog
    18: 1,  # Deer
    19: 1,  # Sheep
    20: 1,  # Cow
    21: 1,  # Elephant
    22: 1,  # Bear
    23: 1,  # Zebra
    24: 1,  # Giraffe
}
annotation_path = os.path.abspath("./archive/FLIR_ADAS_v2/images_thermal_val/coco.json")
data_set_path = os.path.abspath(
    "./evaluation_set/human_and_living_creatures_count_data_set.csv"
)
with open(annotation_path) as f:
    annotation = json.load(f)
length = len(annotation["images"])
data_set = {
    "id": np.zeros(length, dtype=int),
    "file_path": [""] * length,
    "human_count": np.zeros(length, dtype=int),
    "living_creature_count": np.zeros(length, dtype=int),
}
c = 0
for image in annotation["images"]:
    data_set["id"][c] = image["id"]
    path = image["file_name"]
    data_set["file_path"][c] = path
    c += 1
df = pd.DataFrame(data_set)
df.index = df["id"]
df = df.drop(columns=["id"])
for ann in annotation["annotations"]:
    image_id = ann["image_id"]
    category_id = ann["category_id"]
    if category_id in ID_MAPPING:
        class_id = ID_MAPPING[category_id]
        if class_id == 0:
            df.at[image_id, "human_count"] += 1
            df.at[
                image_id, "living_creature_count"
            ] += 1  # Count humans as living creatures too
        elif class_id == 1:
            df.at[image_id, "living_creature_count"] += 1
df.to_csv(data_set_path, index=True)
