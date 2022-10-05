"""
This model applies different post processing steps on the results file of the detectron2 model
"""
import argparse
import csv
import json
import os
from copy import deepcopy
from datetime import date, datetime
from pathlib import Path
from typing import Any, List

import geopy.distance
import numpy as np
import numpy.typing as npt
import pycocotools.mask as mask_util
from panorama.client import PanoramaClient
from shapely.geometry import LineString, Point
from shapely.ops import nearest_points
from tqdm import tqdm
from triangulation.helpers import (
    get_panos_from_points_of_interest,
)  # pylint: disable-all
from triangulation.masking import get_side_view_of_pano
from triangulation.triangulate import triangulate

from visualizations.stats import DataStatistics
from visualizations.utils import get_bridge_information, get_permit_locations

from azure.storage.blob import BlobServiceClient
from azure.identity import ManagedIdentityCredential

client_id = os.getenv("USER_ASSIGNED_MANAGED_IDENTITY")
credential = ManagedIdentityCredential(client_id=client_id)
blob_service_client = BlobServiceClient(account_url="https://cvtdataweuogidgmnhwma3zq.blob.core.windows.net", credential=credential)

def calculate_distance_in_meters(line: LineString, point: Point) -> float:
    """
    Calculates the shortest distance between a line and point, returns a float in meters
    """
    closest_point = nearest_points(line, point)[0]
    return float(geopy.distance.distance(closest_point.coords, point.coords).meters)


def get_container_locations(file: Path) -> List[List[float]]:
    """
    Returns locations of containers from a csv file in lat, lon order
    """
    container_locations = []
    with open(file) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=",")
        next(csv_reader)  # skip first line
        for row in csv_reader:
            container_locations.append([float(row[0]), float(row[1])])
    return container_locations


def save_json_data(data: Any, filename: Path, output_folder: Path) -> None:
    """
    Write the data to a json file
    """
    with open(output_folder / filename, "w") as f:
        json.dump(data, f)


def write_to_csv(data: npt.NDArray[Any], header: List[str], filename: Path) -> None:
    """
    Writes a list of list with data to a csv file.
    """
    np.savetxt(
        filename, data, header=",".join(header), fmt="%d", delimiter=",", comments=""
    )


class PostProcessing:
    """
    Post processing operations on the output of the predictor
    """

    def __init__(
        self,
        json_predictions: str,
        threshold: float = 20000,
        mask_degrees: float = 90,
        date_to_check: datetime = datetime(2021, 3, 17),
        permits_file: str = "decos.xml", # TODO remove
        bridges_file: str = "bridges.geojson", # TODO remove
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


    def write_output(self, output_file_name, cluster_intersections) -> None:
        """
        Write clustered intersections (a 2D list of floats) to an output file.
        """
        num_clusters = cluster_intersections.shape[0]

        print("Number of output ICM clusters: {0:d}".format(num_clusters))

        with open(output_file_name, "w") as inter:
            inter.write("lat,lon\n")
            for i in range(num_clusters):
                inter.write("{0:f},{1:f}\n".format(cluster_intersections[i, 0], cluster_intersections[i, 1]))

        print(f"Done writing cluster intersections to the file: {output_file_name}.")


    def find_points_of_interest(self):
        """
        Finds the points of interest given by COCO format json file, outputs a csv file
        with lon lat coordinates
        """

        if not (self.output_folder / self.data_file).exists():
            save_json_data(self.stats.data, self.data_file, self.output_folder)

        cluster_intersections = triangulate(
            self.output_folder / self.data_file
        )

        return cluster_intersections


    def filter_by_angle(self) -> None:
        """
        Filters the predictions or annotation based on the view angle from the car. If an objects lies within the
        desired angle, then it will be kept
        :return:
        """
        predictions_to_keep = []
        print("Filtering based on angle")
        running_in_k8s = "KUBERNETES_SERVICE_HOST" in os.environ
        for prediction in tqdm(self.stats.data, disable=running_in_k8s):
            response = PanoramaClient.get_panorama(
                prediction["pano_id"].rsplit(".", 1)[0]
            )  # TODO: Not query, look at the database!
            heading = response.heading
            height, width = prediction["segmentation"]["size"]
            mask = get_side_view_of_pano(width, height, heading, self.mask_degrees)[
                :, :, 0
            ]
            # Inverse mask to have region that we would like to keep
            mask = 1 - mask
            segmentation_mask = mask_util.decode(prediction["segmentation"])  # .float()
            overlap = segmentation_mask * mask
            # prediction fully in side view
            if np.sum(overlap) == np.sum(segmentation_mask):
                predictions_to_keep.append(prediction)
            # prediction partially in side view
            # Keep predictions if minimal 50% is in the side view
            elif np.sum(overlap) / np.sum(segmentation_mask) > 0.5:
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
            np.array(
                [
                    prioritized_containers[:, 0],
                    prioritized_containers[:, 1],
                    sorted_scores,
                    permit_distances_sorted,
                    bridges_distances_sorted,
                ]
            ),
            ["lat", "lon", "score", "permit_distance", "bridge_distance"],
            self.output_folder / self.prioritized_file,
        )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run postprocessing for container detection pipeline"
    )
    # TODO remove default
    parser.add_argument("--container_ref_files", type=str, help="TODO", default="postprocessing-input")
    parser.add_argument("--container_detections", type=str, help="TODO", default="detections")
    parser.add_argument("--current_date", type=str, help="Full path to output dir, current date", default="2022-10-03")
    parser.add_argument("--permits_file", type=str, help="Full path to permits file", default="930651BCFAD14D26A3CC96C751CD208E_small.xml")
    parser.add_argument("--bridges_file", type=str, help="Full path to bridges file", default="vuln_bridges.geojson")
    args = parser.parse_args()

    # update output folder inside the docker container
    output_folder = Path(args.current_date)
    if not output_folder.exists():
        output_folder.mkdir(exist_ok=True, parents=True)

    permits_file = f"{args.current_date}/{args.permits_file}"
    predictions_file = f"{args.current_date}/coco_instances_results.json"

    # TODO optimize for loop searching
    # download images from storage account
    container_client = blob_service_client.get_container_client(container=args.container_ref_files)
    blob_list = container_client.list_blobs()
    for blob in blob_list:
        print(f"blob is {blob}")
        path = blob.name
        print(f"path is {path}")

        if path == permits_file or path == args.bridges_file:
            print("trying to open ..")

            with open(f"{blob.name}", "wb") as download_file:
                download_file.write(container_client.get_blob_client(blob).download_blob().readall())

    container_client = blob_service_client.get_container_client(container=args.container_detections)
    blob_list = container_client.list_blobs()
    for blob in blob_list:
        print(f"blob is {blob}")
        path = blob.name
        print(f"path is {path}")

        if path == predictions_file:
            print("trying to open ..")

            with open(f"{blob.name}", "wb") as download_file:
                download_file.write(container_client.get_blob_client(blob).download_blob().readall())
            break

    postprocess = PostProcessing(
        predictions_file,
        output_folder=output_folder,
        permits_file=args.permits_file,
        bridges_file=args.bridges_file,
    )
    postprocess.filter_by_size()
    postprocess.filter_by_angle()
    clustered_intersections = postprocess.find_points_of_interest()
    print(clustered_intersections)
    # postprocess.write_output(os.path.join(args.current_date, "points_of_interest.csv"), clustered_intersections)
    # panoramas = get_panos_from_points_of_interest(
    #     os.path.join(args.current_date, "points_of_interest.csv"),
    #     date(2021, 3, 18),
    #     date(2021, 3, 17),
    # )
    # postprocess.prioritize_notifications()
    # for pano in panoramas:
    #     PanoramaClient.download_image(pano, output_location=args.current_date)

    # TODO remove
    print("downloaded files are")
    print(f"cwd is {os.getcwd()}")
    print(f"ls of files {os.listdir(os.getcwd())}")