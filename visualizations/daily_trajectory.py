"""
This module is responsible for visualizing the trajectory and container found on a day.
Show the containers that were found on the particular trajectory that was driven for a particular day.
"""
import datetime
import json
import re
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Dict, List, Union

import folium
from folium.plugins import MarkerCluster
from panorama import models
from panorama.client import PanoramaClient


def unify_model_output(
    coco_format: Union[Path, str], instances_results: Union[Path, str]
) -> Dict[Any, Any]:
    """
    This method merges information from output files of the model.
    Rationale: The instances results do not have file_name, therefore we cannot map
    predictions to panorama objects. We update the predictions with information about
    file name from the auto-generated coco format file.

    :param coco_format: path to coco format output file
    :param instances_results: path to model predictions/instances results output file

    :returns: instances_results dict with information about file names
    """
    # Opening JSON files
    coco_file = open(coco_format)
    data_format = json.load(coco_file)

    instances_file = open(instances_results)
    predictions: Dict[Any, Any] = json.load(instances_file)

    # get file_name for each prediction
    for prediction in predictions:
        # get id of prediction
        pred_id = prediction["image_id"]
        # extract corresponding file_name from data_format
        file_name = data_format["images"][pred_id]["file_name"]
        # append it to prediction dictionary
        prediction["file_name"] = file_name

    return predictions


def append_prediction_coordinates(predictions: Dict[Any, Any]) -> Dict[Any, Any]:
    """
    This method adds an extra entry in the predictions dicts with information about coordinates.
    Rationale: We want to plot the predictions of the model on a map, therefore we need to retrieve
    information about coordinates from the API.

    :param predictions: model predictions/instances results dict (with information about file names)

    :returns: predictions dict with information about coordinates
    """
    # add dummy coordinates for containers locations for now
    dummy_coords = [[52.352158, 4.908507], [52.359169, 4.906641]]
    for i, prediction in enumerate(predictions):
        prediction["coords"] = dummy_coords[i]

        # TODO: query API for panorama object based on panorama id
        """
        # get panorama_id
        file_name = re.compile(r"[^/]+$")
        pano_id = file_name.search(prediction["file_name"]).group()  

        #query API for panorama object based on panorama id 
        pano_obj = None
        
        # get coordinates
        long, lat, _ = pano_obj.geometry.coordinates
        
        # update predictions dict with coordinates
        prediction["coords"] = [lat, long]
        """

    return predictions


def get_daily_panoramas(target_date: date) -> models.PagedPanoramasResponse:
    """
    This method queries the panorama API for all panorama objects stored at a specific date.

    :param target_date: date we are interested to know trsjectory for
    :returns: paged list of panorama objects based on query

    """
    query_result: models.PagedPanoramasResponse = PanoramaClient.list_panoramas(
        timestamp_after=target_date, timestamp_before=target_date + timedelta(days=1)
    )
    return query_result


def get_panorama_coords(
    daily_panoramas: models.PagedPanoramasResponse,
) -> List[List[float]]:
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

            next_pano_batch: models.PagedPanoramasResponse = PanoramaClient.next_page(
                daily_panoramas
            )
            daily_panoramas = next_pano_batch
        except ValueError:
            print("No next page available")
            break

    return scan_coords


def generate_map(trajectory: List[List[float]], predictions: Dict[Any, Any]) -> None:
    """
    This method generates an HTML page with a map containing a path line and randomly chosen points on the line
    corresponding to detected containers on the path.

    :param trajectory: list of coordinates that define the path.
    :param predictions: model predictions dict (with information about file names and coordinates).
    """
    # Amsterdam coordinates
    latitude = 52.377956
    longitude = 4.897070

    Map = folium.Map(location=[latitude, longitude], zoom_start=12)

    marker_cluster = MarkerCluster().add_to(Map)

    for i in range(0, len(predictions)):
        folium.Marker(
            location=[predictions[i]["coords"][0], predictions[i]["coords"][1]],
            popup="Confidence score: {:.0%}".format(predictions[i]["score"]),
            icon=folium.Icon(
                color="lightgreen",
                icon_color="darkgreen",
                icon="square",
                angle=0,
                prefix="fa",
            ),
            radius=15,
        ).add_to(marker_cluster)

    # Create the map and add the line
    folium.PolyLine(trajectory, color="green", weight=10, opacity=0.8).add_to(Map)

    Map.save("Daily Trajectory.html")


def run(
    day_to_plot: datetime.date,
    coco_format: Union[Path, str],
    instances_results: Union[Path, str],
) -> None:
    """
    This method creates visualization of a path and detected containers based on trajectory on a specific date.

    :param day_to_plot: target date for which we want to see the trajectory and detections.
    :param coco_format: path to coco format output file.
    :param instances_results: path to model predictions/instances results output file.
    """
    # dummy trajectory coordinates which map the dummy containers coodinates
    coords = [
        [52.337909, 4.892184],
        [52.340400, 4.892549],
        [52.340701, 4.892993],
        [52.340570, 4.896835],
        [52.340400, 4.901105],
        [52.340190, 4.905375],
        [52.340374, 4.908250],
        [52.341173, 4.911040],
        [52.342655, 4.912950],
        [52.344136, 4.912863],
        [52.346260, 4.912219],
        [52.348331, 4.910653],
        [52.349654, 4.909859],
        [52.352158, 4.908507],
        [52.354163, 4.906640],
        [52.356221, 4.904988],
        [52.358082, 4.904001],
        [52.358724, 4.903851],
        [52.359169, 4.906641],
        [52.359956, 4.908143],
        [52.360322, 4.908593],
        [52.361161, 4.908143],
    ]

    # for actual trajectory coordinates retrieved from the API uncomment the two lines below
    """
    daily_query_result = get_daily_panoramas(day_to_plot)
    coords = get_panorama_coords(daily_query_result)
    """

    model_predictions = unify_model_output(
        coco_format=coco_format, instances_results=instances_results
    )
    append_prediction_coordinates(model_predictions)
    generate_map(coords, model_predictions)


if __name__ == "__main__":
    target_day = date(2016, 7, 21)
    coco_format_file = "../output/container_train_coco_format.json"
    predictions_file = "../output/coco_instances_results.json"
    run(target_day, coco_format_file, predictions_file)
