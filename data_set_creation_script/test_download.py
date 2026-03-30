import gdown
import os


def download_test_images(output_path=None):
    if output_path is None:
        output_path = os.path.abspath("./")
    os.makedirs(output_path, exist_ok=True)

    url = "https://drive.google.com/drive/folders/14giVFq2z4kLpDT_nEhatNT7AO2h5gIOq?usp=sharing"
    output = os.path.join(
        output_path, "test_images"
    )  # the test images are only 10 so they were not zipped
    gdown.download_folder(url, output, quiet=False)


if __name__ == "__main__":
    download_test_images()
