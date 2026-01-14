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
