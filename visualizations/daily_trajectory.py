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

from panorama import models  # pylint: disable=import-error
from panorama.client import PanoramaClient  # pylint: disable=import-error
from tqdm import tqdm, trange

from visualizations.model import ModelPrediction
from visualizations.unique_instance_prediction import generate_map


def remove_faulty_annotations(annotations):
    correct_images = []
    faulty_ids = []
    for ann in annotations["images"]:
        filename = ann["file_name"]
        is_correct = not filename.split("/")[1].startswith("pano")
        if is_correct:
            correct_images.append(ann)
        else:
            faulty_ids.append(ann["id"])

    annotations["images"] = correct_images

    return annotations, faulty_ids


def remove_corresponding_predictions(predictions, faulty_ids):
    correct_predictions = []
    for pred in predictions:
        is_correct = not pred["image_id"] in faulty_ids
        if is_correct:
            correct_predictions.append(pred)

    return correct_predictions


def unify_model_output(
    coco_annotations: Union[Path, str], instances_results: Union[Path, str]
) -> List[ModelPrediction]:
    """
    This method merges information from output files of the model.
    Rationale: The instances results do not have file_name, therefore we cannot map
    predictions to panorama objects. We update the predictions with information about
    file name from the auto-generated coco format file.

    :param coco_annotations: path to coco annotations file
    :param instances_results: path to model predictions/instances results output file

    :returns: instances_results dict with information about file names
    """
    # Opening JSON files
    coco_file = open(coco_annotations)
    annotations = json.load(coco_file)
    coco_file.close()

    instances_file = open(instances_results)
    predictions_loaded = json.load(instances_file)
    instances_file.close()

    #   TODO remove these 2 lines if the data is in the correct format
    # EXTRA STEP: discard predictions where ann file contains FAULTY image IDs, i.e. filename instead of pano id
    # annotations, faulty_ids = remove_faulty_annotations(annotations)
    # predictions_loaded = remove_corresponding_predictions(predictions_loaded, faulty_ids)

    predictions = []
    # get file_name for each prediction
    for prediction in predictions_loaded:
        # get id of prediction
        pred_id = prediction["image_id"]
        # extract corresponding file_name from annotations
        found = False
        for ann in annotations["images"]:
            if ann["id"] == pred_id:
                if len(ann["file_name"].split("/")) > 1:
                    file_name = ann["file_name"].split("/")[1]
                else:
                    file_name = ann["file_name"]
                # append it to prediction dictionary
                predictions.append(
                    ModelPrediction(filename=file_name, score=prediction["score"])
                )
                found = True
                break

        if found is False:
            raise "No annotation was found"

    return predictions


def append_prediction_coordinates(
    predictions: List[ModelPrediction],
) -> List[ModelPrediction]:
    """
    This method adds an extra entry in the predictions dicts with information about coordinates.
    Rationale: We want to plot the predictions of the model on a map, therefore we need to retrieve
    information about coordinates from the API.

    :param predictions: model predictions/instances results dict (with information about file names)

    :returns: predictions dict with information about coordinates
    """

    for i, prediction in tqdm(enumerate(predictions), total=len(predictions)):
        # query API for panorama object based on panorama id

        # get panorama_id
        pano_id = prediction.filename.split(".")[0]  # remove .jpg extension

        # query API for panorama object based on panorama id
        pano_obj = PanoramaClient.get_panorama(pano_id)

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

    sorted_lists = sorted(zip(timestamps, scan_coords), key=lambda x: x[0])
    sorted_timestamps, sorted_coords = [[x[i] for x in sorted_lists] for i in range(2)]

    return sorted_coords


def run(
    day_to_plot: datetime.date,
    location_query: models.LocationQuery,
    coco_annotations: Union[Path, str],
    instances_results: Union[Path, str],
) -> None:
    """
    This method creates visualization of a path and detected containers based on trajectory on a specific date.

    :param day_to_plot: target date for which we want to see the trajectory and detections.
    :param location_query: location information for API search
    :param coco_annotations: path to coco annotations file.
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

    model_predictions = unify_model_output(
        coco_annotations=coco_annotations, instances_results=instances_results
    )
    # TODO use this variable when model is trained instead of the temporary dummy
    model_predictions_with_coords = append_prediction_coordinates(model_predictions)
    generate_map(trajectory=coords, predictions=model_predictions_with_coords)
    # generate_map(trajectory=coords, predictions=model_predictions)


if __name__ == "__main__":

    target_day = date(2021, 3, 17)

    # Address: Kloveniersburgwal 45
    lat = 52.370670
    long = 4.898990
    # radius = 2000
    radius = 2000
    location_query = models.LocationQuery(latitude=lat, longitude=long, radius=radius)
    coco_val_annotations_file = (
        "../combined/containers-annotated-COCO-test-first-11-batches.json"
    )
    predictions_file = (
        "../outputs/INFER_2kx4k_resolution_1_Mar-27-01:43/coco_instances_results.json"
    )
    run(target_day, location_query, coco_val_annotations_file, predictions_file)
