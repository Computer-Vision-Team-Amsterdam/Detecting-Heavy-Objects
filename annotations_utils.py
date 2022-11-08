
import os
import json
import time
from pathlib import Path
from tqdm import tqdm
from typing import Any, Dict, List, NamedTuple, Tuple, Union

from panorama.client import PanoramaClient

from visualizations.model import PointOfInterest
from visualizations.unique_instance_prediction import geo_clustering, generate_map, color_generator, append_geohash,get_points
from dataclass_wizard import Container


def collect_pano_ids(srcs: List[str], exclude_prefix: str) -> List[str]:
    """
    Given a list of json annotation files, return image filenames.

    Args:
        srcs: list of paths to json files
        exclude: pattern filename to exclude

    return image filenames, excluding @param exclude
    """
    filenames_as_list = []
    for src in tqdm(srcs, total=len(srcs), desc="Collecting pano ids from annotation files"):
        with open(src, "r") as read_file:
            content = json.load(read_file)
        read_file.close()

        images_obj = content["images"]
        filenames_per_src = [os.path.splitext(os.path.basename(image_obj["file_name"]))[0]
                             for image_obj in images_obj]
        print(f"Source {src} has {len(filenames_per_src)} filenames.")
        filenames_as_list.extend(filenames_per_src)

    filenames_TMX = [fn for fn in list(set(filenames_as_list)) if not fn.startswith(exclude_prefix)]
    print(f"There are {len(filenames_TMX)} unique filenames which start with TMX.")

    return filenames_TMX


def get_filenames_metadata(filenames: List[str]) -> Container[PointOfInterest]:
    """

    return map with images_coords
    """
    filenames_loc = Container[PointOfInterest]()
    for filename in tqdm(filenames, total=len(filenames), desc="Getting metadata from API"):
        pano_object = PanoramaClient.get_panorama(filename)
        lat = pano_object.geometry.coordinates[1]
        lon = pano_object.geometry.coordinates[0]

        filenames_loc.append(
            PointOfInterest(
                pano_id=filename,
                coords=(lat, lon)
            )
        )
        # panorama API allows 6 requests per 10 seconds = 1 request per 0.6 seconds
        time.sleep(0.6)

    return filenames_loc


def split_pano_ids(points, nr_clusters, train_ratio=0.7, validation_ratio=0.15):
    """
    Split panorama filenames in train.txt, val.txt and test.txt based on given split ratio
    Panos that start with pano* are firstly moved based on where they already are.
    Example: if pano0001 is initially in annotations-train.json, then we put
    pano0001 in train.txt.
    """

    total = len(points) + 355 # add the pano files

    print(f"total: {total}")
    train_count = int(train_ratio*total)
    val_count = int(validation_ratio*total)
    test_count = total - train_count - val_count

    train_points = []
    val_points = []
    test_points = []

    print(f"Number of panorama files: {len(points)}")
    print(f"Train count is {train_count}, val count is {val_count}, test count is {test_count}.")

    def _not_full(a_list, threshold):
        if len(a_list) < threshold:
            return True
        return False

    for cluster_id in range(nr_clusters):
        points_subset = get_points(points, cluster_id=cluster_id)
        if _not_full(train_points, threshold=train_count - 355):  # we already start with 355 pano* filenames.
            train_points.extend(points_subset)
        elif _not_full(val_points, threshold=val_count):
            val_points.extend(points_subset)
        elif _not_full(test_points, threshold=test_count):
            test_points.extend(points_subset)

    print(f"After filename splitting, train count is {len(train_points)}, val count is {len(val_points)}"
          f" and test count is {len(test_points)}.")

    return train_points, val_points, test_points
