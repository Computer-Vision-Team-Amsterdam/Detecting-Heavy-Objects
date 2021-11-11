import os
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
