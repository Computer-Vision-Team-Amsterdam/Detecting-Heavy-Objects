"""
This module contains general functionality to handle the annotated data
"""
import itertools
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Union

import cv2
import numpy as np
import yaml
from detectron2.structures import BoxMode


def get_container_dicts(img_dir: Union[Path, str]) -> List[Dict[str, Any]]:
    """
    Parse annotations json

    :param img_dir: path to directory with images and annotations
    :return: images metadata
    """

    json_file = os.path.join(img_dir, "containers-annotated.json")
    with open(json_file) as file:
        imgs_anns = json.load(file)
        try:
            imgs_anns = imgs_anns["_via_img_metadata"]
        except KeyError:
            print("Annotation file has no metadata.")

    dataset_dicts = []
    for idx, value in enumerate(imgs_anns.values()):
        record = {}  # type: Dict[str, Any]

        filename = os.path.join(img_dir, value["filename"])
        height, width = cv2.imread(filename).shape[:2]

        record["file_name"] = filename
        record["image_id"] = idx
        record["height"] = height
        record["width"] = width

        annos = value["regions"]
        objs = []
        for anno in annos:
            # assert not anno["region_attributes"]
            anno = anno["shape_attributes"]
            p_x = anno["all_points_x"]
            p_y = anno["all_points_y"]
            poly = [(x + 0.5, y + 0.5) for x, y in zip(p_x, p_y)]
            poly = [p for x in poly for p in x]

            obj = {
                "bbox": [np.amin(p_x), np.amin(p_y), np.amax(p_x), np.amax(p_y)],
                "bbox_mode": BoxMode.XYXY_ABS,
                "segmentation": [poly],
                "category_id": 0,
            }
            objs.append(obj)
        record["annotations"] = objs
        dataset_dicts.append(record)
    return dataset_dicts


def collect_nested_lists(dictionary, composed_key, nested_lists):
    """
    This method parses a nested dictionary recursively and collects the (composed) keys where the value is a list.
    :param dictionary:
    :param composed_key:
    :param nested_lists:

    :return: keys and values of the @dictionary such that values are of type list.
    """

    for k, v in dictionary.items():
        if isinstance(v, dict):
            if composed_key == "":
                collect_nested_lists(v, k, nested_lists)
            else:
                collect_nested_lists(v, composed_key+"."+k, nested_lists)
        if isinstance(v, list):
            if composed_key == "":
                nested_lists[k] = v
            else:
                nested_lists[composed_key+"."+k] = v

    return nested_lists


def generate_config_file(file, configuration, name):

    for composed_name, value in configuration.items():
        names = composed_name.split(".")
        if len(names) == 1:
            file[names[0]] = value
        if len(names) == 2:
            file[names[0]][names[1]] = value
        if len(names) == 3:
            file[names[0]][names[1]][names[2]] = value

    with open(f'configs/temp_{name}.yaml', 'w') as outfile:
        yaml.dump(file, outfile, sort_keys=False)


def handle_hyperparameters(config):

    # open yaml file as dict
    with open(config) as f:
        file = yaml.safe_load(f)

    # get rows for which we do hyperparameter search
    grid_space = collect_nested_lists(file, "", {})

    count = 0
    for combination in itertools.product(*grid_space.values()):
        configuration = dict(zip(grid_space.keys(), combination))
        generate_config_file(file, configuration, count)
        count = count + 1

    return count
