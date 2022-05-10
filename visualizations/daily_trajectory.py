"""
This module is responsible for visualizing the trajectory and container found on a day.
Show the containers that were found on the particular trajectory that was driven for a particular day.
"""
import datetime
import json
import re
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from model import ModelPrediction
from panorama import models  # pylint: disable=import-error
from panorama.client import PanoramaClient  # pylint: disable=import-error
from tqdm import tqdm, trange
from unique_instance_prediction import generate_map


def create_prediction_objects(
    instances_results: Union[Path, str]
) -> List[ModelPrediction]:
    """
    This method create prediction objects with metadata needed for the map
    :param instances_results: path to model predictions/instances results output file

    :returns: instances_results dict with information about pano ids
    """
    # Opening JSON file
    instances_file = open(instances_results)
    predictions_loaded = json.load(instances_file)
    instances_file.close()

    predictions = []
    for prediction in predictions_loaded:
        pano_id = prediction["pano_id"].split(".")[0]
        predictions.append(ModelPrediction(pano_id=pano_id, score=prediction["score"]))

    return predictions


def append_prediction_coordinates(
    predictions: List[ModelPrediction],
) -> List[ModelPrediction]:
    """
    This method adds an extra entry in the predictions dicts with information about coordinates.
    Rationale: We want to plot the predictions of the model on a map, therefore we need to retrieve
    information about coordinates from the API.

    :param predictions: model predictions/instances results dict (with information about panorama ids)

    :returns: predictions dict with information about coordinates
    """

    for i, prediction in tqdm(
        enumerate(predictions),
        total=len(predictions),
        desc="Collect predictions' coords",
    ):

        # query API for panorama object based on panorama id
        pano_obj = PanoramaClient.get_panorama(prediction.pano_id)

        # get coordinates
        long, lat, _ = pano_obj.geometry.coordinates

        # update predictions dict with coordinates
        prediction.coords = [lat, long]

    return predictions


def get_daily_panoramas(
    target_date: date, location_query: models.LocationQuery
) -> models.PagedPanoramasResponse:
    """
    This method queries the panorama API for all panorama objects stored at a specific date.

    :param target_date: date we are interested to know trajectory for
    :param location_query: search query
    :returns: paged list of panorama objects based on query

    """
    query_result: models.PagedPanoramasResponse = PanoramaClient.list_panoramas(
        location=location_query,
        timestamp_after=target_date,
        timestamp_before=target_date + timedelta(days=1),
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

    total_pano_pages = int(daily_panoramas.count / 25)
    print(f"There is a total of {total_pano_pages} panorama pages to iterate over.")
    print(50 * "=")
    pano_page_count = 0
    scan_coords = []
    timestamps = []
    while True:
        pano_page_count = pano_page_count + 1
        if pano_page_count % 20 == 0:
            print(f"Finished {pano_page_count} out of {total_pano_pages}.")
        try:
            for i in range(len(daily_panoramas.panoramas)):
                panorama: models.Panorama = daily_panoramas.panoramas[i]
                long, lat, _ = panorama.geometry.coordinates
                timestamps.append(panorama.timestamp)
                scan_coords.append([lat, long])

            next_pano_batch: models.PagedPanoramasResponse = PanoramaClient.next_page(
                daily_panoramas
            )
            daily_panoramas = next_pano_batch
        except ValueError:
            print("No next page available")
            break

    sorted_lists = sorted(zip(timestamps, scan_coords), key=lambda x: x[0])  # type: ignore
    sorted_timestamps, sorted_coords = [[x[i] for x in sorted_lists] for i in range(2)]

    return sorted_coords


def run(
    day_to_plot: datetime.date,
    location_query: models.LocationQuery,
    instances_results: Union[Path, str],
) -> None:
    """
    This method creates visualization of a path and detected containers based on trajectory on a specific date.

    :param day_to_plot: target date for which we want to see the trajectory and detections.
    :param location_query: location information for API search
    :param instances_results: path to model predictions/instances results output file.
    """
    # dummy trajectory coordinates which map the dummy containers coordinates
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

    daily_query_result = get_daily_panoramas(day_to_plot, location_query)
    coords = get_panorama_coords(daily_query_result)

    model_predictions = create_prediction_objects(instances_results=instances_results)
    model_predictions_with_coords = append_prediction_coordinates(model_predictions)
    generate_map(trajectory=coords, predictions=model_predictions_with_coords)


if __name__ == "__main__":

    target_day = date(2021, 3, 17)

    # Address: Kloveniersburgwal 45
    lat = 52.370670
    long = 4.898990
    radius = 2000
    location_query = models.LocationQuery(latitude=lat, longitude=long, radius=radius)

    predictions_file = (
        "../outputs/INFER_detectron_map2_2_May-09-14:55/coco_instances_results.json"
    )
    run(target_day, location_query, predictions_file)
