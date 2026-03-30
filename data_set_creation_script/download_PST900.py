import os
import dataset_tools as dtools
import zipfile


def download_PST900(output_path=None):
    if output_path is None:
        output_path = os.path.abspath("./real_data")
    os.makedirs(output_path, exist_ok=True)

    output = os.path.join(output_path, "PST900.zip")
    dtools.download(dataset="PST900 RGB-T", dst_dir=output_path)

    with zipfile.ZipFile(output, "r") as zip_ref:
        zip_ref.extractall(output_path)
    os.remove(output)
    print("PST900 dataset downloaded and extracted successfully.")


if __name__ == "__main__":
    download_PST900()
