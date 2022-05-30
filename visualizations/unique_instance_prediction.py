"""
This module implements several methods to remove multiple predictions for the same objects.
It is likely that the same container is detected in multiple images, thus we want to prevent that
we plot/register the same container instance multiple time on the map/result file.
"""
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import folium
import geohash as gh
import pandas as pd
from folium.plugins import MarkerCluster
from model import PointOfInterest
from panorama.client import PanoramaClient


def read_coordinates(decos_file: Union[Path, str]) -> List[PointOfInterest]:
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
        container_loc = PointOfInterest(pano_id="", coords=c)
        # append it to output list
        container_locations.append(container_loc)

    return container_locations


def append_geohash(
    container_locations: List[PointOfInterest],
) -> List[PointOfInterest]:
    """
    This method takes each coordinate pair, computes and stores its geohash alongside with the coordinates.

    :param container_locations: containers latitude, longitudes + metadata such as confidence score

    :returns: container locations with their corresponding geohash
    """

    for container_loc in container_locations:
        # get coordinates
        coords = container_loc.coords
        # compute geohash
        geohash = gh.encode(coords[0], coords[1])
        # store geohash
        container_loc.geohash = geohash

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


def color(cluster_id: int, colors: List[str]) -> str:
    """
    Helper method to assign colors to clusters so we can visually distinguish them.

    :param cluster_id: id of the cluster
    :param colors: sequence of non-unique colors

    :returns: hex code for specific cluster
    """
    return colors[cluster_id]


def generate_map(
    vulnerable_bridges,
    permit_locations,
    trajectory: Optional[List[List[float]]] = None,
    detections: Optional[List[PointOfInterest]] = None,
    name: Optional[str] = None,
    colors: Optional[List[str]] = None,
) -> None:
    """
    This method generates an HTML page with a map containing a path line and randomly chosen points on the line
    corresponding to detected containers on the path.

    :param vulnerable_bridges: list of line string coordinates.
    :param permit_locations: list of point coordinates.
    :param trajectory: list of coordinates that define the path.
    :param detections: model predictions dict (with information about file names and coordinates).
    :param name: custom name for the map. If not passed, name is created based on what the map contains.
    :param colors: colors to be assigned to each cluster
    """
    # Amsterdam coordinates
    latitude = 52.377956
    longitude = 4.897070

    # create empty map zoomed on Amsterdam
    Map = folium.Map(location=[latitude, longitude], zoom_start=12)

    # add container locations to the map
    if detections:
        marker_cluster = MarkerCluster().add_to(Map)  # options={"maxClusterRadius":20}
        for i in range(0, len(detections)):

            # get link to panorama to display

            pano_id = detections[i].pano_id
            image = PanoramaClient.get_panorama(pano_id)
            image_link = image.links.equirectangular_small.href

            # create HTML with more info
            html = (
                f"""
                   <!DOCTYPE html>
                   <html>
                   <center><img src=\""""
                + image_link
                + """\" width=400 height=200 ></center>
                   </html>
                   """
            )
            popup = folium.Popup(folium.Html(html, script=True), max_width=500)
            folium.Marker(
                location=[detections[i].coords[0], detections[i].coords[1]],
                popup=popup,
                icon=folium.Icon(
                    color="lightgreen",
                    icon_color=color(detections[i].cluster, colors)
                    if colors
                    else "darkgreen",
                    icon="square",
                    angle=0,
                    prefix="fa",
                ),
                radius=15,
            ).add_to(marker_cluster)

    # add line with car trajectory on the map
    if trajectory:
        folium.PolyLine(trajectory, color="green", weight=5, opacity=0.8).add_to(Map)

    vulnerable_bridges_group = folium.FeatureGroup(name="Vulnerable bridges").add_to(
        Map
    )

    # add data of vulnerable bridges and canal walls to the map
    for linestring in vulnerable_bridges:
        vulnerable_bridges_group.add_child(
            folium.PolyLine(linestring, color="yellow", weight=5, opacity=0.8).add_to(
                Map
            )
        )

    # add permit locations on the map
    for point in permit_locations:
        folium.CircleMarker(
            location=[point[0], point[1]], color="red", radius=1, weight=5
        ).add_to(Map)

    folium.LayerControl().add_to(Map)

    # create name for the map
    if not name:
        if detections and trajectory:
            name = "Daily trajectory and predicted containers"
        if detections and not trajectory:
            name = "Daily predicted containers"
        if not detections and trajectory:
            name = "Daily trajectory"
        if not detections and not trajectory:
            name = "Empty map"

    Map.save(f"{name}.html")


def geo_clustering(
    container_locations: List[PointOfInterest], prefix_length: int
) -> Tuple[List[PointOfInterest], int]:
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
        geohash = container_loc.geohash
        geo_prefix = geohash[:prefix_length]
        if geo_prefix in unique_prefixes:
            container_loc.cluster = unique_prefixes[geo_prefix]
        else:
            unique_prefixes[geo_prefix] = cluster_id
            container_loc.cluster = cluster_id
            cluster_id = cluster_id + 1

    nr_clusters = cluster_id

    return container_locations, nr_clusters


if __name__ == "__main__":
    container_metadata = read_coordinates("../decos/Decos.xlsx")
    container_metadata_with_geohash = append_geohash(container_metadata)
    container_metadata_clustered, total_clusters = geo_clustering(
        container_metadata_with_geohash, prefix_length=5
    )

    generate_map(
        detections=container_metadata_clustered,
        name="Decos containers",
        colors=color_generator(total_clusters),
    )
