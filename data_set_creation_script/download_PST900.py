import gdown
import os
import zipfile


def download_PST900(path=None):
    if path is None:
        path = os.path.abspath("./real_data")
    os.makedirs(path, exist_ok=True)
    url = f"https://drive.google.com/file/d/1hZeM-MvdUC_Btyok7mdF00RV-InbAadm/view"
    output = os.path.join(path, "PST900.zip")
    gdown.download(url, output, quiet=False, fuzzy=True)
    if not os.path.exists(output):
        raise Exception(f"Failed to download the dataset. from {url}")
    with zipfile.ZipFile(output, "r") as zip_ref:
        zip_ref.extractall(path)
    os.remove(output)


if __name__ == "__main__":
    download_PST900()
