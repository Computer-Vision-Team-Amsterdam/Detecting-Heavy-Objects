"""
This module implements several methods to remove multiple predictions for the same objects.
It is likely that the same container is detected in multiple images, thus we want to prevent that
we plot/register the same container instance multiple time on the map/result file.
"""
import random
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

import geohash as gh
import pandas as pd

from visualizations.daily_trajectory import generate_map


def read_coordinates(decos_file: Union[Path, str]) -> List[Dict[str, List[float]]]:
    """
    This method reads data from Decos.xlsx. We run the clustering algorithm on the geocoordinates from Decos
    until we have a trained model whose output coordinates we can use.

    :param decos_file: path to Decos file

    :returns: latitude and longitude of all containers in decos.
    """

    container_locations = []
    data = pd.read_excel(decos_file)
    coordinates = data[["LATITUDE", "LONGITUDE"]].values.tolist()

    for c in coordinates:
        # create dict
        # we store the coordinate in this format to be consistent with detectron output files.
        container_loc = {"coords": c, "score": 0.85}
        # append it to output list
        container_locations.append(container_loc)

    return container_locations


def append_geohash(
    container_locations: List[Dict[str, List[float]]]
) -> List[Dict[str, List[float]]]:
    """
    This method takes each coordinate pair, computes and stores its geohash alongside with the coordinates.

    :param container_locations: containers latitude, longitudes + metadata such as confidence score

    :returns: container locations with their corresponding geohash
    """

    for container_loc in container_locations:
        # get coordinates
        coords = container_loc["coords"]
        # compute geohash
        geohash = gh.encode(coords[0], coords[1])
        # store geohash
        container_loc["geohash"] = geohash

    return container_locations


def color_generator(nr_colors: int) -> List[str]:
    """
    This method returns a sequence of random, not strictly unique, colors.

    :param nr_colors: number of colors to be generated.

    :return: sequence of hex codes
    """
    colors = [
        "#" + "".join([random.choice("0123456789ABCDEF") for _ in range(6)])
        for _ in range(nr_colors)
    ]

    return colors


def geo_clustering(
    container_locations: List[Dict[str, Any]], prefix_length: int
) -> Tuple[List[Dict[str, Any]], int]:
    """
    This method looks at all container geocodes and clusters them based on the first prefix_length digits.
    For example: We have 2 geocodes u173yffw8qjy and u173yffvndbb.
                If prefix_length is 8 or greater, they do not belong to the same cluster.
                If prefix_length is 7 or smaller, then they belong to the same cluster

    :param container_locations: container latitude, longitudes, geohash + metadata such as confidence score
    :param prefix_length: length of the common geohash prefix.
    """

    if prefix_length < 0 or prefix_length > 12:
        raise ValueError("Prefix must be an integer in [0, 12] interval.")

    unique_prefixes: Dict[str, int] = {}
    cluster_id = 0
    for container_loc in container_locations:
        geohash = container_loc["geohash"]
        geo_prefix = geohash[:prefix_length]
        if geo_prefix in unique_prefixes:
            container_loc["cluster"] = unique_prefixes[geo_prefix]
        else:
            unique_prefixes[geo_prefix] = cluster_id
            container_loc["cluster"] = cluster_id
            cluster_id = cluster_id + 1

    nr_clusters = cluster_id

    return container_locations, nr_clusters


if __name__ == "__main__":
    container_metadata = read_coordinates("../Decos.xlsx")
    container_metadata_with_geohash = append_geohash(container_metadata)
    container_metadata_clustered, total_clusters = geo_clustering(
        container_metadata_with_geohash, prefix_length=5
    )

    generate_map(
        predictions=container_metadata_clustered,
        name="Decos containers",
        colors=color_generator(total_clusters),
    )
