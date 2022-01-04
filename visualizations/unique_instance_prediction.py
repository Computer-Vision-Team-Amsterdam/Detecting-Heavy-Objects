"""
This module implements several methods to remove multiple predictions for the same objects.
It is likely that the same container is detected in multiple images, thus we want to prevent that
we plot/register the same container instance multiple time on the map/result file.
"""
from pathlib import Path
from typing import Union, List, Dict

import pandas as pd

from daily_trajectory import generate_map


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


def naive_clustering(radius):
    """
    This method looks at all container locations and defines  treats each cluster of containers as a single instance i
    """
    pass


if __name__ == "__main__":
    coords = read_coordinates("../Decos.xlsx")
    generate_map(predictions=coords, name="Decos containers")