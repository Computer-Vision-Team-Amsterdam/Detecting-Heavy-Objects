"""
This model applies different post processing steps on the results file of the detectron2 model
"""
import argparse
import csv
import json
import os
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any, List

import geopy.distance
import numpy as np
import numpy.typing as npt
import pandas as pd
import pandas.io.sql as sqlio
import pycocotools.mask as mask_util
from psycopg2.extras import execute_values
from scipy.spatial.distance import cdist
from shapely.geometry import LineString, Point
from shapely.ops import nearest_points
from triangulation.masking import get_side_view_of_pano
from triangulation.triangulate import triangulate

import upload_to_postgres
from utils.azure_storage import StorageAzureClient
from utils.date import get_start_date
from visualizations.model import PointOfInterest
from visualizations.stats import DataStatistics
from visualizations.unique_instance_prediction import generate_map
from visualizations.utils import get_bridge_information, get_permit_locations


def closest_point(point: float, points: List[float]) -> Any:
    """
    Find closest point from a list of points.
    """
    return points[cdist([point], points).argmin()]


def match_value(df: Any, col1: str, x: float, col2: str) -> Any:
    """
    Match value x from col1 row to value in col2.
    """
    return df[df[col1] == x][col2].values[0]


def get_closest_pano(df: Any, clustered_intersections: Any) -> Any:
    """
    Find a panorama closest to a found container (intersection point).
    """
    df["point"] = [
        (x, y) for x, y in zip(df["camera_location_lat"], df["camera_location_lon"])
    ]

    closest_points = [
        closest_point(x, list(df["point"])) for x in clustered_intersections[:, :2]
    ]
    pano_match = [match_value(df, "point", x, "file_name") for x in closest_points]

    # Flatten the list
    return np.concatenate(pano_match).ravel()

def get_closest_pano2(df: Any, clustered_intersections: Any) -> Any:
    """
    Find a panorama closest to a found container (intersection point).
    """
    df["point"] = [
        (x, y) for x, y in zip(df["camera_location_lat"], df["camera_location_lon"])
    ]

    closest_points = [
        closest_point(x, list(df["point"])) for x in clustered_intersections[:, :2]
    ]
    pano_match = [match_value(df, "point", x, "file_name") for x in closest_points]

    # Flatten the list
    return pano_match

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
    Write the data to a json file.
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
        fmt="%1.6f,%1.6f,%1.6f,%1.6f,%1.6f,%s,%s",
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

    def find_points_of_interest(self) -> Any:
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
        for prediction in self.stats.data:
            heading = (
                query_df["heading"]
                .loc[query_df["file_name"] == prediction["pano_id"]]
                .values[0]
            )

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
        self, panoramas: List[str], container_locations: List[float]
    ) -> Any:

        """
        Prioritize all found containers based on the permits and locations compared to the vulnerable bridges and
        canals. This function returns a structured array (column based) with floats and strings inside.
        """

        def calculate_score(bridge_distance: float, permit_distance: float) -> float:
            """
            Calculate score for bridge and permit distance;
            High score --> Permit big, bridge  small
            medium score --> Permit big, bridge big or permit small, bridge big
            low score --> permit small, bridge big
            """
            if permit_distance >= 25:
                return 1 + max([(25 - bridge_distance) / 25, 0])
            else:
                return 0

        permit_locations, permit_keys, permit_locations_failed = get_permit_locations(
            self.permits_file, self.date_to_check
        )

        bridge_locations = get_bridge_information(self.bridges_file)

        container_locations_geom = [Point(location) for location in container_locations]
        permit_locations_geom = [
            Point(permit_location) for permit_location in permit_locations
        ]
        bridge_locations_geom = [
            LineString(bridge_location)
            for bridge_location in bridge_locations
            if bridge_location
        ]

        # Failed to parse some items in the permits file
        np.savetxt(
            "permit_locations_failed.csv",
            permit_locations_failed,
            header="DECOS_ITEM_KEY",
            fmt="%s",
            delimiter=",",
            comments="",
        )

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
        closest_permits = []
        for container_location in container_locations_geom:
            bridge_container_distances = []
            for bridge_location in bridge_locations_geom:
                try:
                    bridge_dist = calculate_distance_in_meters(bridge_location, container_location)
                except:
                    bridge_dist = 10000
                    print("Error occured:")
                    print(f"Container location: {container_location}, {container_location.coords}")
                    print(f"Bridge location: {bridge_location}")
                bridge_container_distances.append(bridge_dist)
            closest_bridge_distance = min(bridge_container_distances)
            bridges_distances.append(round(closest_bridge_distance, 2))

            closest_permit_distances = []
            for permit_location in permit_locations:
                try:
                    permit_dist = geopy.distance.distance(container_location.coords, permit_location.coords).meters
                except:
                    permit_dist = 0
                    print("Error occured:")
                    print(f"Container location: {container_location}, {container_location.coords}")
                    print(f"Permit location: {permit_location}, {permit_location.coords}")
                closest_permit_distances.append(permit_dist)
            permit_distances.append(np.amin(closest_permit_distances))
            closest_permits.append(permit_keys[np.argmin(closest_permit_distances)])
        scores = [
            calculate_score(bridges_distances[idx], permit_distances[idx])
            for idx in range(len(container_locations))
        ]

        print(f"Closest permits: {closest_permits}")
        sorted_indices = np.argsort([score * -1 for score in scores])
        prioritized_containers = np.array(container_locations)[sorted_indices]
        permit_distances_sorted = np.array(permit_distances)[sorted_indices]
        bridges_distances_sorted = np.array(bridges_distances)[sorted_indices]
        permit_keys = np.array(closest_permits)[sorted_indices]
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
                    permit_keys,
                )
            ),
            dtype=[
                ("lat", float),
                ("lon", float),
                ("score", float),
                ("permit_distance", float),
                ("bridge_distance", float),
                ("closest_image", "U64"),
                ("permit_keys", "U64"),
            ],
        )

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
        help="Processing date in the format %Y-%m-%d %H:%M:%S.%f",
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
    parser.add_argument(
        "--permits_file",
        type=str,
        help="Full path to permits file",
        default="decos_dump.xml",
    )
    parser.add_argument(
        "--bridges_file",
        type=str,
        help="Full path to bridges file",
        default="vuln_bridges.geojson",
    )
    args = parser.parse_args()

    start_date_dag, start_date_dag_ymd = get_start_date(args.date)

    # Update output folder inside the WORKDIR of the docker container
    output_folder = Path(start_date_dag_ymd)
    if not output_folder.exists():
        output_folder.mkdir(exist_ok=True, parents=True)

    permits_file = f"{start_date_dag_ymd}/{args.permits_file}"

    # Get access to the Azure Storage account.
    azure_connection = StorageAzureClient(secret_key="data-storage-account-url")

    # Get all prediction files where the mission day is the same the start_date
    all_blobs = azure_connection.list_container_content(cname=args.bucket_detections)
    same_day_coco_jsons = [
        blob
        for blob in all_blobs
        if blob.split("/")[0].startswith(start_date_dag_ymd)
        and blob.split("/")[-1].endswith(".json")
    ]

    # Download all coco_instances_results.json from the same day
    for blob in same_day_coco_jsons:
        blob_date = blob.split("/")[0]
        local_file_path = Path(blob_date)
        if not local_file_path.exists():
            local_file_path.mkdir(exist_ok=True, parents=True)
        azure_connection.download_blob(
            cname=args.bucket_detections, blob_name=blob, local_file_path=blob
        )
        print(f"Downloaded {blob}")

    combined_predictions = []
    for blob in same_day_coco_jsons:
        f = open(blob)
        predictions = json.load(f)
        combined_predictions.extend(predictions)
        f.close()

    combined_detection_results = "coco_instances_results_combined.json"
    # Save combined json file in WORKDIR of the Docker container
    with open(combined_detection_results, "w") as outfile:
        json.dump(combined_predictions, outfile)

    # Download files to the WORKDIR of the Docker container.
    azure_connection.download_blob(
        cname=args.bucket_ref_files,
        blob_name=permits_file,
        local_file_path=permits_file,
    )

    azure_connection.download_blob(
        cname=args.bucket_ref_files,
        blob_name=args.bridges_file,
        local_file_path=args.bridges_file,
    )

    # Find possible object intersections from detections in panoramic images.
    postprocess = PostProcessing(
        Path(combined_detection_results),  # TODO why use Path
        output_folder=output_folder,
        date_to_check=datetime.strptime(start_date_dag_ymd, "%Y-%m-%d"),
        permits_file=permits_file,
        bridges_file=args.bridges_file,
    )

    with upload_to_postgres.connect() as (conn, cur):
        table_name = "containers"

        # Get images with a detection
        # TODO: perform sanitizing inputs SQL. For now, code will break if start_date_dag_ymd is not a datetime.
        sql = (
            f"SELECT B.file_name, B.heading, B.camera_location_lat, B.camera_location_lon FROM detections A "
            f"LEFT JOIN images B ON A.file_name = B.file_name WHERE "
            f"date_trunc('day', taken_at) = '{start_date_dag_ymd}'::date;"
        )

        query_df = sqlio.read_sql_query(sql, conn)
        query_df = pd.DataFrame(query_df)

        if query_df.empty:
            print(
                "DataFrame is empty! No images with a detection are found for the provided date."
            )
        else:
            postprocess.filter_by_size()
            postprocess.filter_by_angle()
            clustered_intersections = postprocess.find_points_of_interest()
            print(clustered_intersections)

            # Get columns
            sql = f"SELECT * FROM {table_name} LIMIT 0"
            cur.execute(sql)
            table_columns = [desc[0] for desc in cur.description]
            table_columns.pop(0)  # Remove the id column

            # Find a panorama closest to an intersection
            clustered_intersections = clustered_intersections[:, :2]
            try:
                pano_match = get_closest_pano(query_df, clustered_intersections)
            except:
                pano_match = get_closest_pano2(query_df, clustered_intersections)

            pano_match_prioritized = postprocess.prioritize_notifications(
                pano_match, clustered_intersections
            )

            vulnerable_bridges = get_bridge_information(postprocess.bridges_file)
            (
                permit_locations,
                permit_keys,
                permit_locations_failed,
            ) = get_permit_locations(permits_file, postprocess.date_to_check)

            # Create maps
            detections = []
            for row in pano_match_prioritized:
                lat, lon, score, _, _, closest_image, permit_key = row
                closest_permit = permit_locations[permit_keys.index(permit_key)]
                detections.append(
                    PointOfInterest(
                        pano_id=closest_image.split(".")[0],  # remove .jpg
                        coords=(float(lat), float(lon)),
                        closest_permit=(closest_permit[0], closest_permit[1]),
                        score=score,
                    )
                )

            # Create overview map
            generate_map(
                vulnerable_bridges,
                permit_locations,
                trajectory=None,
                detections=detections,
                name="Overview",
            )

            # Create prioritized map
            generate_map(
                vulnerable_bridges,
                permit_locations,
                trajectory=None,
                detections=detections[:25],
                name="Prioritized",
            )

            # Insert the values in the database
            sql = f"INSERT INTO {table_name} ({','.join(table_columns)}) VALUES %s"
            # we don't want permit_keys in the database.
            cols_to_insert = list(pano_match_prioritized.dtype.names)[:-1]
            execute_values(cur, sql, pano_match_prioritized[cols_to_insert])
            conn.commit()

            # Upload the file with found containers to the Azure Blob Storage
            for csv_file in ["prioritized_objects.csv", "permit_locations_failed.csv"]:
                azure_connection.upload_blob(
                    "postprocessing-output",
                    os.path.join(start_date_dag, csv_file),
                    csv_file,
                )
            # Upload the combined json file to the Azure Blob Storage
            azure_connection.upload_blob(
                "postprocessing-output",
                os.path.join(start_date_dag, combined_detection_results),
                combined_detection_results,
            )

            # Upload overview and prioritized maps to the Azure Blob Storage
            for html_file in ["Overview.html", "Prioritized.html"]:
                azure_connection.upload_blob(
                    "postprocessing-output",
                    os.path.join(start_date_dag, html_file),
                    html_file,
                )