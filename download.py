import argparse
import os
from pathlib import Path

from panorama import models
from panorama.client import PanoramaClient

from tqdm import tqdm

from historical_container_locations import download_images


def download_text_file(root, file):

    filename = Path(file).stem
    with open(Path(root, f"{file}.txt")) as f:
        pano_ids = f.read().splitlines()
    panorama_images = []
    for pano_id in tqdm(pano_ids, desc="Collecting panoramas"):
        panorama_image = PanoramaClient.get_panorama(pano_id)
        panorama_images.append(panorama_image)

    output_location = f"data_azure/{filename}/images"
    download_images(panorama_images, models.ImageSize.MEDIUM, output_location)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--f', type=str, default="1",
                        help='Text file number to download')
    parser.add_argument('--root', type=str, default="data_azure/pano_ids")
    args = parser.parse_args()

    download_text_file(args.root, args.f)

    # create classes.txt file used for labelImg
    filename = f"data_azure/{args.f}/images/classes.txt"
    print(f"Created classes.txt file in {filename}")
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, "w") as f:
        f.write("person\nlicence_plate")
    f.close()


