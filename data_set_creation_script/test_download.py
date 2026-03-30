import gdown
import os


path = os.path.abspath("./")


def download_test_images():
    url = "https://drive.google.com/drive/folders/14giVFq2z4kLpDT_nEhatNT7AO2h5gIOq?usp=sharing"
    output = os.path.join(
        path, "test_images"
    )  # the test images are only 10 so they were not zipped
    gdown.download_folder(url, output, quiet=False)
