"""
This module finds and downloaded the panoramas from crowded places in Amsterdam.

Pipeline explanation:
- We define the crowded locations in crowded_locations.csv
- We query the panorama API to find the panorama ids of these crowded locations.
- We query CloudVPS with the panorama ids in order to download the panorama images.
"""

import csv
from datetime import timedelta
from multiprocessing import Pool
from typing import List

from panorama import models
from panorama.client import PanoramaClient

from blurring.download_pano_from_cloudvps import download_image


def get_panorama_ids(query_result: models.PagedPanoramasResponse):

    """
    Get list of panorama ids based on query result
    """
    if len(query_result.panoramas) == 0:
        raise ValueError("No available panoramas.")

    pano_ids = []
    total_pano_pages = int(query_result.count / 25)
    pano_page_count = 0
    while True:
        pano_page_count = pano_page_count + 1
        if pano_page_count % 20 == 0:
            print(f"Finished {pano_page_count} out of {total_pano_pages}.")
        try:
            for i in range(len(query_result.panoramas)):
                panorama: models.Panorama = query_result.panoramas[i]
                id = panorama.id
                pano_ids.append(id)
            next_pano_batch: models.PagedPanoramasResponse = PanoramaClient.next_page(
                query_result
            )
            query_result = next_pano_batch
        except ValueError:
            print("No next page available")
            break

    return pano_ids


def query_API(row: List):
    """
    Query API based on location, date, radius from the crowded_locations.csv
    """
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

    return query_result


if __name__ == "__main__":
    all_crowded_panoramas_ids = []

    # read CSV file with list of crowded locations
    with open("crowded_locations.csv", 'r') as file:
        reader = csv.reader(file)

    # loop through the crowded locations
    for row in reader:

        # find the panorama objects found at the crowded locations
        crowded_panorama_objs = query_API(row)

        # extract the ids of the panoramas
        crowded_panorama_ids = get_panorama_ids(crowded_panorama_objs)

        # append the ids to the complete list of ids
        all_crowded_panoramas_ids.extend(crowded_panorama_ids)

    # write crowded panorama ids to CSV file
    with open('crowded_pano_ids.csv', 'w') as myfile:
        wr = csv.writer(myfile, delimiter='\n')
        wr.writerow(all_crowded_panoramas_ids)

    # download the panoramas from crowded locations from CloupVPS
    p = Pool(8)
    p.map(download_image, all_crowded_panoramas_ids)