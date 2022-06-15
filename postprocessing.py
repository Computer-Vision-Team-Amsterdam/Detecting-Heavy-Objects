"""
This model applies different post processing steps on the results file of the detectron2 model
"""
import csv
from copy import deepcopy
from datetime import date, datetime
from pathlib import Path
from typing import List

import geojson
import geopy.distance
import numpy as np
import pycocotools.mask as mask_util
import torch
from panorama.client import PanoramaClient
from shapely.geometry import LineString, Point
from tqdm import tqdm
from triangulation.helpers import (
    get_panos_from_points_of_interest,
)  # pylint: disable-all
from triangulation.masking import get_side_view_of_pano
from triangulation.triangulate import triangulate

from osgeo import osr  # pylint: disable-all

from utils import (
    calculate_distance_in_meters,
    get_container_locations,
    get_permit_locations,
    save_json_data,
    write_to_csv,
)
from visualizations.stats import DataStatistics



class PostProcessing:
    """
    Post processing operations on the output of the predictor
    """

    def __init__(
        self,
        json_predictions: Path,
        threshold: float = 20000,
        mask_degrees: float = 90,
        date_to_check: datetime = datetime(2021, 3, 17),
        permits_file: Path = Path("decos.xml"),
        bridges_file: Path = Path("bridges.geojson"),
        output_folder: Path = Path.cwd(),
    ) -> None:
        """
        Args:
            :param json_predictions: path to ouput file with predictions from detectron2
            :param threshold: objects with a bbox smaller and equal to this arg are discarded. value is in pixels
            :param mask_degrees: Area from side of the car, eg 90 means that on both sides of the 90 degrees is kept
            :param output_folder: where the filtered json with predictions is stored
            :
        """
        self.stats = DataStatistics(json_file=json_predictions)
        self.non_filtered_stats = deepcopy(self.stats)
        self.threshold = threshold
        self.mask_degrees = mask_degrees
        self.output_folder = output_folder
        self.permits_file = permits_file
        self.bridges_file = bridges_file
        self.date_to_check = date_to_check
        self.output_folder.mkdir(exist_ok=True, parents=True)
        self.objects_file = Path("points_of_interest.csv")
        self.data_file = Path("processed_predictions.json")
        self.prioritized_file = Path("prioritized_objects.csv")

    def find_points_of_interest(self) -> None:
        """
        Finds the points of interest given by COCO format json file, outputs a csv file
        with lon lat coordinates
        """

        if not (self.output_folder / self.data_file).exists():
            save_json_data(self.stats.data, self.data_file, self.output_folder)

        triangulate(
            self.output_folder / self.data_file,
            self.output_folder / self.objects_file,
        )

    def filter_by_angle(self) -> None:
        """
        Filters the predictions or annotation based on the view angle from the car. If an objects lies within the
        desired angle, then it will be kept
        :return:
        """
        predictions_to_keep = []
        print("Filtering based on angle")
        for prediction in tqdm(self.stats.data):
            response = PanoramaClient.get_panorama(
                prediction["pano_id"].rsplit(".", 1)[0]
            )  # TODO: Not query, look at the database!
            heading = response.heading
            height, width = prediction["segmentation"]["size"]
            mask = torch.from_numpy(
                get_side_view_of_pano(width, height, heading, self.mask_degrees)
            )[
                :, :, 0
            ]  # mask is 3D
            mask = (
                1 - mask.float()
            )  # Inverse mask to have region that we would like to keep
            segmentation_mask = torch.from_numpy(
                mask_util.decode(prediction["segmentation"])
            ).float()
            overlap = segmentation_mask * mask
            # prediction fully in side view
            if torch.sum(overlap) == torch.sum(segmentation_mask):
                predictions_to_keep.append(prediction)
            # prediction partially in side view
            # Keep predictions if minimal 50% is in the side view
            elif torch.sum(overlap) / torch.sum(segmentation_mask) > 0.5:
                predictions_to_keep.append(prediction)
        print(
            f"{len(self.stats.data) - len(predictions_to_keep)} out of the {len(self.stats.data)} are filtered based on the angle"
        )
        self.stats.update(predictions_to_keep)

    def filter_by_size(self) -> None:
        """
        Removes predictions of small objects and writes results to json file
        """
        indices_to_keep = [
            idx for idx, area in enumerate(self.stats.areas) if area > self.threshold
        ]
        print(
            f"{len(self.stats.data) - len(indices_to_keep)} out of the {len(self.stats.data)} are filtered based on the size"
        )

        self.stats.update([self.stats.data[idx] for idx in indices_to_keep])

    def prioritize_notifications(self) -> None:
        """
        Prioritize all found containers based on the permits and locations compared to the vulnerable bridges and canals
        """

        def calculate_score(bridge_distance: float, permit_distance: float) -> float:
            """
            Calculate score for bridge and permit distance;
            High score --> Permit big, bridge  small
            medium score --> Permit big, bridge big or permit small, bridge big
            low score --> permit small, bridge big
            """
            return permit_distance + max([25 - bridge_distance, 0])

        permit_locations = get_permit_locations(self.permits_file, self.date_to_check)
        permit_locations_geom = [
            Point(permit_location) for permit_location in permit_locations
        ]
        bridge_locations = get_bridge_information(self.bridges_file)
        bridge_locations_geom = [
            LineString(bridge_location)
            for bridge_location in bridge_locations
            if bridge_location
        ]
        container_locations = get_container_locations(
            self.output_folder / self.objects_file
        )
        container_locations_geom = [Point(location) for location in container_locations]

        bridges_distances = []
        permit_distances = []
        for container_location in container_locations_geom:
            closest_bridge_distance = min(
                [
                    calculate_distance_in_meters(bridge_location, container_location)
                    for bridge_location in bridge_locations_geom
                ]
            )
            bridges_distances.append(closest_bridge_distance)

            closest_permit_distance = min(
                [
                    geopy.distance.distance(
                        container_location.coords, permit_location.coords
                    ).meters
                    for permit_location in permit_locations_geom
                ]
            )
            permit_distances.append(closest_permit_distance)
        scores = [
            calculate_score(bridges_distances[idx], permit_distances[idx])
            for idx in range(len(container_locations))
        ]
        sorted_indices = np.argsort([score * -1 for score in scores])
        prioritized_containers = np.array(container_locations)[sorted_indices]
        permit_distances_sorted = np.array(permit_distances)[sorted_indices]
        bridges_distances_sorted = np.array(bridges_distances)[sorted_indices]
        sorted_scores = np.array(scores)[sorted_indices]

        write_to_csv(
            [
                prioritized_containers[:, 0],
                prioritized_containers[:, 1],
                sorted_scores,
                permit_distances_sorted,
                bridges_distances_sorted,
            ],
            ["lat", "lon", "score", "permit_distance", "bridge_distance"],
            self.output_folder / self.prioritized_file,
        )

def get_bridge_information(file: Union[Path, str]) -> List[List[List[float]]]:
    """
    Return a list of coordinates where to find vulnerable bridges and canal walls
    """

    def rd_to_wgs(coordinates: List[float]) -> List[float]:
        """
        Convert rijksdriehoekcoordinates into WGS84 cooridnates. Input parameters: x (float), y (float).
        """
        epsg28992 = osr.SpatialReference()
        epsg28992.ImportFromEPSG(28992)

        epsg4326 = osr.SpatialReference()
        epsg4326.ImportFromEPSG(4326)

        rd2latlon = osr.CoordinateTransformation(epsg28992, epsg4326)
        lonlatz = rd2latlon.TransformPoint(coordinates[0], coordinates[1])
        return [float(value) for value in lonlatz[:2]]

    bridges_coords = []
    with open(file) as f:
        gj = geojson.load(f)
    features = gj["features"]
    print("Parsing the bridges information")
    for feature in tqdm(features):
        bridge_coords = []
        if feature["geometry"]["coordinates"]:
            for idx, coords in enumerate(feature["geometry"]["coordinates"][0]):
                bridge_coords.append(rd_to_wgs(coords))
            # only add to the list when there are coordinates
            bridges_coords.append(bridge_coords)
    return bridges_coords


if __name__ == "__main__":
    output_path = Path("Test")
    postprocess = PostProcessing(
        Path("coco_instances_results_14ckpt.json"), output_folder=output_path
    )
    postprocess.filter_by_size()
    postprocess.filter_by_angle()
    postprocess.find_points_of_interest()
    panoramas = get_panos_from_points_of_interest(
        output_path / "points_of_interest.csv", date(2021, 3, 18), date(2021, 3, 17)
    )
    postprocess.prioritize_notifications()
    for pano in panoramas:
        PanoramaClient.download_image(pano, output_location=output_path)
