"""
This module contains general functionality to handle the annotated data
"""
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Union

import cv2
import numpy as np
from detectron2.structures import BoxMode


def get_container_dicts(img_dir: Union[Path, str]) -> List[Dict[str, Any]]:
    """
    Parse annotations json

    :param img_dir: path to directory with images and annotations
    :return: images metadata
    """

    json_file = os.path.join(img_dir, "containers-annotated.json")
    with open(json_file) as f:
        imgs_anns = json.load(f)
        imgs_anns = imgs_anns["_via_img_metadata"]

    dataset_dicts = []
    for idx, v in enumerate(imgs_anns.values()):
        record = {}  # type: Dict[str, Any]

        filename = os.path.join(img_dir, v["filename"])
        height, width = cv2.imread(filename).shape[:2]

        record["file_name"] = filename
        record["image_id"] = idx
        record["height"] = height
        record["width"] = width

        annos = v["regions"]
        objs = []
        for anno in annos:
            # assert not anno["region_attributes"]
            anno = anno["shape_attributes"]
            px = anno["all_points_x"]
            py = anno["all_points_y"]
            poly = [(x + 0.5, y + 0.5) for x, y in zip(px, py)]
            poly = [p for x in poly for p in x]

            obj = {
                "bbox": [np.amin(px), np.amin(py), np.amax(px), np.amax(py)],
                "bbox_mode": BoxMode.XYXY_ABS,
                "segmentation": [poly],
                "category_id": 0,
            }
            objs.append(obj)
        record["annotations"] = objs
        dataset_dicts.append(record)
    return dataset_dicts
