"""This module contains functionality to search for images containers based on
the Decos dataset and evaluate the quality of the search"""
import argparse
import glob
import os
import random
import shutil
from datetime import date
from pathlib import Path
from typing import List, Tuple, Union

import pandas as pd
from panorama import models  # pylint: disable=import-error
from panorama.client import PanoramaClient  # pylint: disable=import-error
from panorama.models import Panorama  # pylint: disable=import-error
from tqdm import tqdm


class Coordinate:
    """
    Class to encode the geolocation of a panorama
    """
    def __init__(self, latitude: float, longitude: float) -> None:
        self.latitude = latitude
        self.longitude = longitude


def parse_file_from_decos(
    path_to_file: Union[Path, str],
) -> Tuple[List[Coordinate], List[Tuple[date, date]]]:
    """
    This function looks at the csv/json file from Decos
    and returns list of coords and the datetime
    """
    data = pd.read_excel(path_to_file)

    coords = []
    intervals = []
    for _, row in tqdm(data.iterrows(), total=len(data), desc="Parsing Decos file..."):
        start, end, long, lat = (
            row["DATUM_VAN"],
            row["DATUM_TM"],
            row["LONGITUDE"],
            row["LATITUDE"],
        )
        coords.append(Coordinate(longitude=long, latitude=lat))
        intervals.append((start, end))

    return coords, intervals


def filter_panoramas(
    coordinates: List[Coordinate], radius: int, time_intervals: List[Tuple[date, date]]
) -> List[Panorama]:
    """
    This method queries the Panorama API for images based on list of locations and times

    :param coordinates: latitudes and longitudes for each panorama we are interested in
    :param radius: area of search around the coordinates. All images with coordinates
                   within radius are retrieved
    :param time_intervals: start and end period for each coordinate.

    returns: List of len(coordinates) panorama objects

    """
    query_results = []
    for i in tqdm(range(len(coordinates)), desc="Filtering images from API..."):
        coord = coordinates[i]
        location = models.LocationQuery(
            latitude=coord.latitude, longitude=coord.longitude, radius=radius
        )

        date_after, date_before = time_intervals[i]
        timestamp_after = date(date_after.year, date_after.month, date_after.day)
        timestamp_before = date(date_before.year, date_before.month, date_before.day)
        try:
            query_result: models.PagedPanoramasResponse = PanoramaClient.list_panoramas(
                location=location,
                timestamp_after=timestamp_after,
                timestamp_before=timestamp_before,
            )
        except TimeoutError:
            print(f"Time after {timestamp_after}, time before {timestamp_before}")

        query_results.extend(query_result.panoramas)

    return query_results


def download_images(
    images: List[Panorama], image_size: str, output_location: Union[Path, str]
) -> None:
    """
    This method downloads a list of images to a specified path.

    :param images: panorama objects to be downloaded
    :param image_size: the size to download the images in. Options: SMALL, MEDIUM, FULL
    :param output_location: path to the directory where the downloaded images are stored

    """
    os.makedirs(output_location, exist_ok=True)
    for image in tqdm(images, desc="Downloading panoramas"):
        if os.path.isfile(Path(output_location, image.filename)):
            print(f"{image.filename} was already downloaded. Overwriting...")
        try:
            PanoramaClient.download_image(
                image, size=image_size, output_location=output_location
            )
        except TimeoutError:
            print("Timed out downloading operation...Skipping.")


def find_container_images(radius: int) -> List[Panorama]:
    """
    Pipeline that searches and stores container images starting from
    Decos permit requests.
    """
    coordinates, time_intervals = parse_file_from_decos("Decos.xlsx")
    images = filter_panoramas(coordinates, radius, time_intervals)
    print(f"Query returned {len(images)} results.")

    return images


def evaluate_images_filtering(
    target_path: Union[Path, str], subset_size: int
) -> None:
    """
    Helper method to visualize sample of images that are queried from the panorama API.

    :param target_path: path to the directory where the downloaded images are stored
    :param subset_size: number of images to visualize from the output_location
    """

    filtered_panorama_files = glob.glob(f"{target_path}/*.jpg")

    if len(filtered_panorama_files) < subset_size:
        raise Exception("The sample size exceeds the number of available images.")
    sample = random.sample(filtered_panorama_files, subset_size)

    print(f"INFO: Creating folder with subset of {subset_size} images...")
    temporary_dir = Path(output_location, "temporary")
    os.makedirs(temporary_dir)

    for file in sample:
        shutil.copy(file, temporary_dir)
    os.system(f"labelImg {temporary_dir}")

    print(f"INFO: Deleting folder with subset of {subset_size} images...")
    shutil.rmtree(temporary_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--radius', type=int, default=100,
                        help='Radius of search given lat and long')

    args = parser.parse_args()
    images = find_container_images(args.radius)

    output_location = f"pano_id_filter_radius_{args.radius}"
    download_images(images, models.ImageSize.MEDIUM, output_location)

    evaluate_images_filtering(f"filter_radius_{args.radius}", subset_size=100)
