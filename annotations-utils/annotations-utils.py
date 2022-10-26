import os
import json
from pathlib import Path
from typing import Any, Dict, List, NamedTuple, Tuple, Union
from contextlib import contextmanager

from panorama.client import PanoramaClient

from visualizations.model import PointOfInterest
from visualizations.unique_instance_prediction import geo_clustering, generate_map, color_generator, append_geohash

"""
@contextmanager
def custom_open(filename):
    f = json.load(filename)
    try:
        yield f
    finally:
        f.close()
"""


def collect_pano_ids(srcs: List[str], exclude_prefix: str) -> List[str]:
    """
    Given a list of json annotation files, return image filenames.

    Args:
        srcs: list of paths to json files
        exclude: pattern filename to exclude

    return image filenames, excluding @param exclude
    """
    filenames_as_list = []
    for src in srcs:
        with open(src, "r") as read_file:
            content = json.load(read_file)
        read_file.close()

        images_obj = content["images"]
        filenames_per_src = [os.path.splitext(os.path.basename(image_obj["file_name"]))[0]
                             for image_obj in images_obj]
        filenames_as_list.extend(filenames_per_src)

    filenames_TMX = [fn for fn in list(set(filenames_as_list)) if not fn.startswith(exclude_prefix)]

    return filenames_TMX


def get_filenames_metadata(filenames: List[str]):
    """

    return map with images_coords
    """
    filenames_loc = []
    for filename in filenames:
        pano_object = PanoramaClient.get_panorama(filename)
        lat = pano_object.geometry.coordinates[1]
        lon = pano_object.geometry.coordinates[0]

        filenames_loc.append(
            PointOfInterest(
                pano_id=filename,
                coords=(lat, lon)
            )
        )

    return filenames_loc


def main():
    srcs = ["tests/test_annotations-utils/annotations-as-aml.json",
            "tests/test_annotations-utils/containers-annotated-COCO-val.json"]
    file_names = collect_pano_ids(srcs, exclude_prefix="pano")
    points = get_filenames_metadata(file_names)
    points_geohash = append_geohash(points)
    clustered_points, nr_clusters = geo_clustering(container_locations=points_geohash,
                                                   prefix_length=6)
    colors = color_generator(nr_colors=nr_clusters)
    generate_map(vulnerable_bridges=[],
                 permit_locations=[],
                 detections=clustered_points,
                 name="Clustered_annotations",
                 colors=colors)


main()

