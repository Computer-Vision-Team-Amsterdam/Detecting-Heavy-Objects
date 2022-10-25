"""
This model applies different post processing steps on the results file of the detectron2 model
"""
import argparse
import csv
import json
import os
from copy import deepcopy
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, List

import geopy.distance
import numpy as np
import numpy.typing as npt
import pandas as pd
import pandas.io.sql as sqlio
import pycocotools.mask as mask_util
from panorama.client import PanoramaClient
from psycopg2.extras import execute_values
from scipy.spatial.distance import cdist
from shapely.geometry import LineString, Point
from shapely.ops import nearest_points
from tqdm import tqdm
from triangulation.masking import get_side_view_of_pano
from triangulation.triangulate import triangulate

import upload_to_postgres
from azure_storage_utils import BaseAzureClient, StorageAzureClient
from visualizations.stats import DataStatistics
from visualizations.utils import get_bridge_information, get_permit_locations

azClient = BaseAzureClient()
USERNAME = azClient.get_secret_value("postgresUsername")
USERNAME = f"{USERNAME}@cvt-weu-psql-o-01-silnc2achvsfi"
PASSWORD = azClient.get_secret_value("postgresPassword-short")
HOST = azClient.get_secret_value("postgresHostname")
PORT = "5432"
DATABASE = "container-detection-database"


def closest_point(point, points):
    """Find closest point from a list of points."""
    return points[cdist([point], points).argmin()]


def match_value(df, col1, x, col2):
    """Match value x from col1 row to value in col2."""
    return df[df[col1] == x][col2].values[0]


def get_closest_pano(dat, clustered_intersections):
    df1 = pd.DataFrame(dat)
    df1["point"] = [
        (x, y) for x, y in zip(df1["camera_location_lat"], df1["camera_location_lon"])
    ]

    closest_points = [
        closest_point(x, list(df1["point"])) for x in clustered_intersections[:, :2]
    ]
    pano_match = [match_value(df1, "point", x, "file_name") for x in closest_points]
    return np.concatenate(pano_match).ravel() # Flatten the list


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


def write_to_csv(data: npt.NDArray[Any], filename: Path) -> None:
    """
    Writes a list of list with data to a csv file.
    """
    np.savetxt(
        filename,
        data,
        header=",".join(data.dtype.names),
        fmt="%1.6f,%1.6f,%1.6f,%1.6f,%1.6f,%s",
        delimiter=",",
        comments="",
    )


class PostProcessing:
    """
    Post processing operations on the output of the predictor
    """

    def __init__(
        self,
        json_predictions: Path,
        date_to_check: datetime,
        permits_file: str,
        bridges_file: str,
        threshold: float = 20000,
        mask_degrees: float = 90,
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

    def write_output(self, output_file_name: str, cluster_intersections: Any) -> None:
        """
        Write clustered intersections (a 2D list of floats) to an output file.
        """
        num_clusters = cluster_intersections.shape[0]

        print("Number of output ICM clusters: {0:d}".format(num_clusters))

        with open(output_file_name, "w") as inter:
            inter.write("lat,lon\n")
            for i in range(num_clusters):
                inter.write(
                    "{0:f},{1:f}\n".format(
                        cluster_intersections[i, 0], cluster_intersections[i, 1]
                    )
                )

        print(f"Done writing cluster intersections to the file: {output_file_name}.")

    def find_points_of_interest(self) -> Any:  # TODO Any to list of floats
        """
        Finds the points of interest given by COCO format json file, outputs a nested list with lon lat coordinates
        and a score. For example: [[52.32856949, 4.85737839, 2.0]]
        """

        if not (self.output_folder / self.data_file).exists():
            save_json_data(self.stats.data, self.data_file, self.output_folder)

        cluster_intersections = triangulate(self.output_folder / self.data_file)

        return cluster_intersections

    def filter_by_angle(self) -> None:
        """
        Filters the predictions or annotation based on the view angle from the car. If an objects lies within the
        desired angle, then it will be kept
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
        Removes predictions of small objects and updates variable stats
        """
        indices_to_keep = [
            idx for idx, area in enumerate(self.stats.areas) if area > self.threshold
        ]
        print(
            f"{len(self.stats.data) - len(indices_to_keep)} out of the {len(self.stats.data)} are filtered based on the size"
        )

        self.stats.update([self.stats.data[idx] for idx in indices_to_keep])

    def prioritize_notifications(
        self, panoramas: List[str], write_file: bool = False
    ) -> List:
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

        if (
            not bridge_locations_geom
            or not container_locations_geom
            or not permit_locations_geom
        ):
            print(
                "WARNING! an empty list, please check the permit, bridge and container files."
            )

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
        sorted_panoramas = np.array(panoramas)[sorted_indices]

        structured_array = np.array(
            list(
                zip(
                    prioritized_containers[:, 0],
                    prioritized_containers[:, 1],
                    sorted_scores,
                    permit_distances_sorted,
                    bridges_distances_sorted,
                    sorted_panoramas,
                )
            ),
            dtype=[
                ("lat", float),
                ("lon", float),
                ("score", float),
                ("permit_distance", float),
                ("bridge_distance", float),
                ("closest_image", "U64"),
            ],
        )

        if write_file:
            write_to_csv(
                structured_array,
                self.prioritized_file,
            )

        return structured_array


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run postprocessing for container detection pipeline"
    )
    parser.add_argument(
        "--date",
        type=str,
        help="Processing date in the format YYYY-MM-DD",
    )
    parser.add_argument(
        "--bucket_ref_files",
        type=str,
        help="Azure Blob Storage with reference files.",
        default="postprocessing-input",
    )
    parser.add_argument(
        "--bucket_detections",
        type=str,
        help="Azure Blob Storage with predictions file.",
        default="detections",
    )
    # TODO remove default args
    parser.add_argument(
        "--permits_file",
        type=str,
        help="Full path to permits file",
        default="930651BCFAD14D26A3CC96C751CD208E_small.xml",
    )
    # TODO remove default args
    parser.add_argument(
        "--bridges_file",
        type=str,
        help="Full path to bridges file",
        default="vuln_bridges.geojson",
    )
    args = parser.parse_args()

    # Update output folder inside the WORKDIR of the docker container
    output_folder = Path(args.date)
    if not output_folder.exists():
        output_folder.mkdir(exist_ok=True, parents=True)

    permits_file = f"{args.date}/{args.permits_file}"
    predictions_file = f"{args.date}/coco_instances_results.json"

    # Get access to the Azure Storage account.
    azure_connection = StorageAzureClient(secret_key="data-storage-account-url")

    # Download files to the WORKDIR of the Docker container.
    azure_connection.download_blob(args.bucket_ref_files, permits_file, permits_file)
    azure_connection.download_blob(
        args.bucket_ref_files, args.bridges_file, args.bridges_file
    )
    azure_connection.download_blob(
        args.bucket_detections, predictions_file, predictions_file
    )

    postprocess = PostProcessing(
        Path(predictions_file),  # TODO why use Path
        output_folder=output_folder,
        date_to_check=datetime(2021, 3, 17),
        permits_file=permits_file,
        bridges_file=args.bridges_file,
    )

    # Find possible object intersections from detections in panoramic images.
    postprocess.filter_by_size()
    postprocess.filter_by_angle()
    clustered_intersections = postprocess.find_points_of_interest()
    postprocess.write_output(
        os.path.join(args.date, "points_of_interest.csv"), clustered_intersections
    )

    # Make a connection to the database
    conn, cur = upload_to_postgres.connect()
    table_name = "containers"

    # Get columns
    sql = f"SELECT * FROM {table_name}"
    cur.execute(sql)
    table_columns = [desc[0] for desc in cur.description]
    table_columns.pop(0)  # Remove the id column

    # Get images with a detection
    sql = (
        f"SELECT * FROM detections A LEFT JOIN images B ON A.file_name = B.file_name WHERE "
        f"date_trunc('day', taken_at) = '{args.date}'::date;"
    )
    df_pano_det = sqlio.read_sql_query(sql, conn)
    print(df_pano_det)  # TODO check if records are found

    # Find a panorama closest to an intersection
    pano_match = get_closest_pano(df_pano_det, clustered_intersections)

    pano_match_prioritized = postprocess.prioritize_notifications(pano_match)

    # Insert the values in the database
    sql = f"INSERT INTO {table_name} ({','.join(table_columns)}) VALUES %s"
    execute_values(cur, sql, pano_match_prioritized)
    conn.commit()  # TODO verschil commit en execute?

    if conn:
        cur.close()
        conn.close()
        print("PostgreSQL connection is closed")

    # Upload the file with found containers to the Azure Blob Storage
    azure_connection.upload_blob(
        "postprocessing-output",
        os.path.join(args.date, "prioritized_objects.csv"),
        "prioritized_objects.csv",
    )