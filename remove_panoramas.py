import glob
import json
import os
from pathlib import Path


def remove_panoramas(
    src: Path = Path("outputs/empty_predictions.json"),
    path_to_panoramas: Path = Path("outputs/to_delete"),
) -> None:
    """
    Delete panoramas from the storage account after processing based on list of panorama ids.


    :param src: path to json file with names of images to be removed
    :param path_to_panoramas: path to storage account where panoramas are stored.
    """
    print(f"Contents of folder before deletion: {os.listdir(path_to_panoramas)}")

    with open(src) as f:
        entries = json.load(f)
    f.close()

    images_to_delete = [entry["pano_id"] for entry in entries]

    print(f"Number of images to be deleted: {len(images_to_delete)}.")

    img_path_to_delete = glob.glob(f"{path_to_panoramas}/*.jpg")
    for img_path in img_path_to_delete:
        image_name = img_path.split("/")[-1]
        if image_name in images_to_delete:
            os.remove(img_path)

    print(f"Contents of folder after deletion: {os.listdir(path_to_panoramas)}")


if __name__ == "__main__":
    remove_panoramas()
