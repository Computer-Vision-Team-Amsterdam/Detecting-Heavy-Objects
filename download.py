import argparse
import json
import os
import glob
import time
from pathlib import Path
from typing import Dict, Any

from panorama import models
from panorama.client import PanoramaClient

from tqdm import tqdm
import datetime

from historical_container_locations import download_images


def generate_metadata_file(path_to_folder: Path) -> None:
    """
    Create metadata json file named "metadata.json" given a list of panoramas
    :param path_to_folder: path to folder with images
    """

    def json_default(value) -> Dict[Any, Any]:
        """
        Custom json encoder for Panorama object
        """
        if isinstance(value, datetime.datetime):
            return dict(year=value.year,
                        month=value.month,
                        day=value.day,
                        hour=value.hour,
                        minute=value.minute,
                        second=value.second)
        else:
            return value.__dict__

    metadata = []
    for image in glob.glob(f"{path_to_folder}/*.jpg"):
        panorama_id: str = Path(image).stem
        panorama_object:  models.Panorama = PanoramaClient.get_panorama(panorama_id)
        metadata.append(panorama_object)
        time.sleep(10)  # we are allowed to do 6 API requests per 10 seconds

    with open('metadata.json', 'w') as file:
        json.dump(metadata, file, default=json_default)
    file.close()


def batch_pano_ids():
    """
    This method creates batches of 100 files with panorama ids based on query result.
    Here the query is all panoramas which have been taken in the center of the city on 17th of March 2021
    since on this day the car drove through the canals and we are interested in the containers placed nearby.
    """
    # Address: Kloveniersburgwal 45
    lat = 52.370670
    long = 4.898990
    radius = 2000

    timestamp_after = datetime.date(2021, 3, 17)
    timestamp_before = datetime.date(2021, 3, 18)

    root = "data_azure/17mar2021/batches"
    Path(root).mkdir(parents=True, exist_ok=True)  # create folder structure if it doesn't exist yet

    location = models.LocationQuery(
        latitude=lat, longitude=long, radius=radius
    )

    query_results: models.PagedPanoramasResponse = PanoramaClient.list_panoramas(
        location=location,
        timestamp_after=timestamp_after,
        timestamp_before=timestamp_before,
    )
    count = 0
    pano_ids = []
    while True:

        if len(pano_ids) >= 100:

            # save to txt
            with open(f'{root}/{count}.txt', 'w+') as f:
                f.write('\n'.join(pano_ids))

            # increase counter for filename
            count = count + 1

            # reset pano_ids
            pano_ids = []

        for i in range(len(query_results.panoramas)):
            pano_id = query_results.panoramas[i].id
            pano_ids.append(pano_id)

        try:
            next_pano_batch: models.PagedPanoramasResponse = PanoramaClient.next_page(query_results)
            query_results = next_pano_batch
        except ValueError:
            print("No next panorama page.")
            break


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



