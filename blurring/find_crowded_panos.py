"""
This module finds the panoramas from crowded places in Amsterdam.
"""
import csv
from datetime import timedelta
from pathlib import Path

from panorama import models
from panorama.client import PanoramaClient


def read_csv(file: Path):
    """
    Read the csv file with crowded locations
    """
    with open(file, 'r') as file:
        reader = csv.reader(file)
        next(reader)  # skip first line
        for row in reader:

            name, lat, long, radius, date = row
            location = models.LocationQuery(
                latitude=float(lat), longitude=float(long), radius=int(radius)
            )
            if date == "-":
                query_result: models.PagedPanoramasResponse = PanoramaClient.list_panoramas(
                    location=location)
            else:
                query_result: models.PagedPanoramasResponse = PanoramaClient.list_panoramas(
                    location=location,
                    timestamp_after=date,
                    timestamp_before=date + timedelta(days=1))


if __name__ == "__main__":
    read_csv(Path("crowded_locations.csv"))