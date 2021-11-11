import fnmatch
import os
import shutil
from datetime import date
from typing import List
from unittest import TestCase

from panorama import models
from panorama.client import PanoramaClient

from historical_container_locations import Coordinate, download_images, filter_panoramas


class Test(TestCase):

    # TODO: Add way to assert corrected of this method
    def test_filter_panoramas(self) -> None:
        coordinates = list()
        radius = 10
        time_intervals = list()

        coord_0 = Coordinate(52.3626770908732, 4.90774612505295)
        coord_1 = Coordinate(52.3626770908732, 4.90774612505295)
        coord_2 = Coordinate(52.3696770908732, 4.99774612505295)
        time_interval_0 = (date(2018, 1, 1), date(2020, 1, 10))
        time_interval_1 = (date(2016, 2, 1), date(2018, 2, 10))
        time_interval_2 = (date(2019, 3, 1), date(2020, 3, 10))

        coordinates.append(coord_0)
        coordinates.append(coord_1)
        coordinates.append(coord_2)
        time_intervals.append(time_interval_0)
        time_intervals.append(time_interval_1)
        time_intervals.append(time_interval_2)

        actual_query_result = filter_panoramas(coordinates, radius, time_intervals)
        print(f"Number of found panoramas: {len(actual_query_result)}")

    def test_download_images(self) -> None:
        output_location = "test_download_images"
        try:
            # create output folder for the images
            os.makedirs(output_location)
        except FileExistsError:
            # if folder exists it must be deleted before running the test again
            if len(os.listdir(output_location)) == 0:
                os.rmdir(output_location)
            else:
                shutil.rmtree(output_location)

        response: models.PagedPanoramasResponse = PanoramaClient.list_panoramas()
        panoramas: List[models.Panorama] = response.panoramas[:10]
        download_images(
            panoramas,
            image_size=models.ImageSize.MEDIUM,
            output_location=output_location,
        )

        expected_image_count = len(panoramas)
        actual_image_count = len(fnmatch.filter(os.listdir(output_location), "*.jpg"))
        self.assertEqual(
            expected_image_count,
            actual_image_count,
            f"The output folder contains {actual_image_count} images, but it was expected to contain "
            f"{expected_image_count} images.",
        )
