# import some common libraries
import json
import os
import random
from pathlib import Path
from typing import Any, Dict, List, Union

import cv2
import numpy as np
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.engine import DefaultPredictor
from detectron2.structures import BoxMode
from detectron2.utils.logger import setup_logger
from detectron2.utils.visualizer import ColorMode, Visualizer
from matplotlib import pyplot

from inference import CONTAINER_DETECTION_MODEL

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
                "bbox": [np.amin(px), np.amin(py), np.amax(px), np.amax(py)],
                "bbox_mode": BoxMode.XYXY_ABS,
                "segmentation": [poly],
                "category_id": 0,
            }
            objs.append(obj)
        record["annotations"] = objs
        dataset_dicts.append(record)
    return dataset_dicts


def visualize_images(
    dataset_dicts: List[Dict[str, Any]],
    metadata: Dict[Any, Any],
    mode: str,
    n_sample: int,
) -> None:
    """
    Visualize the annotations of randomly selected samples in the dataset.
    Additionally, you can specify a trained model to display the confidence score for each annotation

    :param dataset_dicts: images metadata
    :param mode: the type of visualization, i.e. whether to view the object annotations or the confidence scores.
                 For the second option, there must be a trained model specified in the configurations.
                 Options: [ann, pred]
    :param n_sample: number of samples to be visualized
    :
    """

    for d in random.sample(dataset_dicts, n_sample):
        img = cv2.imread(d["file_name"])
        visualizer = Visualizer(
            img[:, :, ::-1],
            metadata=metadata,
            scale=0.5,
            instance_mode=ColorMode.IMAGE_BW,
        )
        out: DefaultPredictor = None
        if mode == "pred":
            outputs = CONTAINER_DETECTION_MODEL(img)
            out = visualizer.draw_instance_predictions(outputs["instances"].to("cpu"))
        if mode == "ann":
            out = visualizer.draw_dataset_dict(d)
        pyplot.imshow(out.get_image()[:, :, ::-1])
        pyplot.show()


def register_dataset(name: str) -> None:
    """
    Update detectron2 dataset catalog with our custom dataset.
    """

    for d in ["train", "val"]:
        DatasetCatalog.register(
            f"{name}_" + d, lambda d=d: get_container_dicts("data/" + d)
        )
        MetadataCatalog.get(f"{name}_" + d).set(thing_classes=[f"{name}"])


if __name__ == "__main__":

    dataset_name = "container"
    register_dataset(name=dataset_name)
    metadata = MetadataCatalog.get(f"{dataset_name}_train")
    dataset_dicts = get_container_dicts("data/train")
    visualize_images(dataset_dicts, metadata, mode="ann", n_sample=3)
