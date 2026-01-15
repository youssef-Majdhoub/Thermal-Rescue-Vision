import os
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from PIL import Image
import psutil
import tqdm


class data_set_manager(Dataset):
    def __init__(
        self, csv_file, path="./archive/FLIR_ADAS_v2/images_thermal_train", to_GPU=False
    ):
        self.data_frame = pd.read_csv(os.path.abspath(csv_file))
        self.path = os.path.abspath(path)
        if to_GPU and torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        self.data = {}
        for idx in tqdm.tqdm(self.data_frame.index, desc="Loading data"):
            file_path = os.path.join(self.path, self.data_frame.at[idx, "file_path"])
            image = Image.open(file_path).convert("L").resize((640, 512))
            image_tensor = (
                torch.frombuffer(image.tobytes(), dtype=torch.uint8)
                .view(1, image.height, image.width)
                .to(self.device)
            )
            self.data[idx] = [
                image_tensor,
                self.data_frame.at[idx, "human_count"],
                self.data_frame.at[idx, "living_creature_count"],
            ]

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        output = torch.tensor(self.data[idx][1:], dtype=torch.float).to(self.device)
        input_tensor = self.data[idx][0].float() / 255.0
        return input_tensor, output

    def get_memory_usage(self):
        process = psutil.Process(os.getpid())
        mem_info = process.memory_info()
        return mem_info.rss / (1024 * 1024)  # Return memory usage in MB


class data_set_server(DataLoader):
    def __init__(self, data_set, batch_size=24, shuffle=True, num_workers=0):
        super().__init__(
            dataset=data_set,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
        )
