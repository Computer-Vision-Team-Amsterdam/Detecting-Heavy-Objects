import glob
import os
import random
import shutil
from datetime import date
from pathlib import Path
from typing import List, Tuple, Union

from panorama import models
from panorama.client import PanoramaClient
from panorama.models import Panorama


class Coordinate:
    def __init__(self, latitude: float, longitude: float) -> None:
        self.latitude = latitude
        self.longitude = longitude


def parse_file_from_Decos() -> Tuple[List[Coordinate], int, List[Tuple[date, date]]]:
    # TODO: implement this when we get data from Decos
    """
    This function looks at the csv/json file from Decos
    and returns list of coords, an avg radius or search and the datetime
    """
    pass


def filter_panoramas(
    coordinates: List[Coordinate], radius: int, time_intervals: List[Tuple[date, date]]
) -> List[Panorama]:
    """
    This method queries the Panorama API for images based on list of locations and times.

    :param coordinates: latitudes and longitudes for each panorama we are interested in
    :param radius: area of search around the coordinates. All images with coordinates within radius are retrieved
    :param time_intervals: start and end period for each coordinate.

    returns: List of len(coordinates) panorama objects

    """
    query_results = list()
    for i in range(len(coordinates)):
        coord = coordinates[i]
        location = models.LocationQuery(
            latitude=coord.latitude, longitude=coord.longitude, radius=radius
        )

        date_after, date_before = time_intervals[i]
        timestamp_after = date(date_after.year, date_after.month, date_after.day)
        timestamp_before = date(date_before.year, date_before.month, date_before.day)

        query_result: models.PagedPanoramasResponse = PanoramaClient.list_panoramas(
            location=location,
            timestamp_after=timestamp_after,
            timestamp_before=timestamp_before,
        )

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
    for image in images:
        if os.path.isfile(Path(output_location, image.filename)):
            print(f"{image.filename} was already downloaded. Overwriting...")
        PanoramaClient.download_image(
            image, size=image_size, output_location=output_location
        )


def find_container_images() -> None:
    """
    Pipeline that searches and stores container images starting from Decos permit requests.
    """
    coordinates, radius, time_intervals = parse_file_from_Decos()
    images = filter_panoramas(coordinates, radius, time_intervals)

    # TODO: find more meaningful name for output folder. We will compare several versions against each other
    output_location = f"filter_radius_{radius}"
    download_images(images, models.ImageSize.MEDIUM, output_location)


def evaluate_images_filtering(
    output_location: Union[Path, str], subset_size: int
) -> None:
    """
    Helper method to visualize a sample of images that are queried from the panorama API.

    :param output_location: path to the directory where the downloaded images are stored
    :param subset_size: number of images to visualize from the output_location
    """

    filtered_panorama_files = glob.glob(f"{output_location}/*.jpg")

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
