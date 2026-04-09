import gdown
import os
import zipfile
import shutil


def download_falling_human_dataset(path=None):
    if path is None:
        path = os.path.abspath("./")
    os.makedirs(path, exist_ok=True)
    url = "https://drive.google.com/file/d/1doVgA11POy8XxRMvjy6WdHu98MMQ09rd/view?usp=sharing"
    output = os.path.join(path, "falling humans.zip")
    gdown.download(url, output, quiet=False, fuzzy=True)
    if not os.path.exists(output):
        raise Exception(f"Failed to download the dataset. from {url}")
    with zipfile.ZipFile(output, "r") as zip_ref:
        zip_ref.extractall(path)
    os.remove(output)


if __name__ == "__main__":
    download_falling_human_dataset()
