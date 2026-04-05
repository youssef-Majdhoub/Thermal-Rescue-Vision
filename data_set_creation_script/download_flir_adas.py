import kagglehub
import os
import shutil


def download_flir_adas_dataset(output_path=None):
    if output_path is None:
        output_path = os.path.abspath("./archive")
        os.makedirs(output_path, exist_ok=True)
    path = kagglehub.dataset_download(
        "samdazel/teledyne-flir-adas-thermal-dataset-v2",
        output_dir=output_path,
    )
    file_name = os.path.basename(path)
    file_path = os.path.dirname(path)
    print("Path to dataset files:", file_path, "\nDownloaded file:", file_name)
    return path


if __name__ == "__main__":
    download_flir_adas_dataset()
