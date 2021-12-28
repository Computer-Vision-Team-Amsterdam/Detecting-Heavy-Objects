"""
This module is responsible for visualizing the trajectory and container found on a day.
Show the containers that were found on the particular trajectory that was driven for a particular day.
"""

import folium
from panorama.client import PanoramaClient
from panorama import models
from datetime import date, timedelta


def get_daily_panoramas(target_date: date) -> models.PagedPanoramasResponse:
    """
    This method queries the panorama API for all panorama objects stored at a specific date

    :returns:

    """
    query_result: models.PagedPanoramasResponse = PanoramaClient.list_panoramas(
        timestamp_after=target_date,
        timestamp_before=target_date + timedelta(days=1),
        limit_results=1000
    )
    return query_result


def get_panorama_coords(daily_panoramas: models.PagedPanoramasResponse):
    """
    This method collects the coordinates of the panorama objects stored at a specific date
    such that their timestamps are in chronological order

    :returns: list of lists of [latitude, longitude]
    """
    if len(daily_panoramas.panoramas) == 0:
        raise ValueError("No available panoramas.")

    scan_coords = []
    while True:
        try:
            for i in range(len(daily_panoramas.panoramas)):
                panorama: models.Panorama = daily_panoramas.panoramas[i]

                long, lat, _ = panorama.geometry.coordinates
                scan_coords.append([lat, long])
                #print(f"Timestamp: {panorama.timestamp}")

            next_pano_batch: models.PagedPanoramasResponse = PanoramaClient.next_page(daily_panoramas)
            daily_panoramas = next_pano_batch
        except ValueError:
            print("No next page available")
            break

    return scan_coords

def create_map(coordinates):
    # Amsterdam coordindates
    latitude = 52.377956
    longitude = 4.897070

    Map = folium.Map(location=[latitude, longitude],
                     zoom_start=12)

    # Create the map and add the line
    folium.PolyLine(coordinates,
                    color='green',
                    weight=10,
                    opacity=0.8).add_to(Map)

    Map.save('Daily Trajectory.html')


if __name__ == "__main__":
    date = date(2018, 1, 2)
    res = get_daily_panoramas(date)
    coords = get_panorama_coords(res)
    create_map(coords)