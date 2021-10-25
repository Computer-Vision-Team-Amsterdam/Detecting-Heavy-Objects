# import some common libraries
import json
import os
import random
from pathlib import Path
from typing import Any, Dict, List, Union

import cv2
import numpy as np
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.structures import BoxMode
from detectron2.utils.logger import setup_logger
from detectron2.utils.visualizer import Visualizer
from matplotlib import pyplot

# import some common detectron2 utilities


setup_logger()


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
                "bbox": [np.min(px), np.min(py), np.max(px), np.max(py)],
                "bbox_mode": BoxMode.XYXY_ABS,
                "segmentation": [poly],
                "category_id": 0,
            }
            objs.append(obj)
        record["annotations"] = objs
        dataset_dicts.append(record)
    return dataset_dicts


def check_data_loading(dataset_dicts: List[Dict[str, Any]]) -> None:
    """
    Visualize the annotations of randomly selected samples in the training set

    :param dataset_dicts: images metadata

    """
    for d in random.sample(dataset_dicts, 3):
        img = cv2.imread(d["file_name"])
        visualizer = Visualizer(img[:, :, ::-1], metadata=container_metadata, scale=0.5)
        out = visualizer.draw_dataset_dict(d)
        pyplot.imshow(out.get_image()[:, :, ::-1])
        pyplot.show()


def register_dataset() -> None:
    """
    Update detectron2 dataset catalog with our custom dataset.
    """
    for d in ["train", "val"]:
        DatasetCatalog.register(
            "container_" + d, lambda d=d: get_container_dicts("data/" + d)
        )
        MetadataCatalog.get("container_" + d).set(thing_classes=["data"])


if __name__ == "__main__":

    register_dataset()
    container_metadata = MetadataCatalog.get("container_train")
    dataset_dicts = get_container_dicts("data/train")
    check_data_loading(dataset_dicts)
