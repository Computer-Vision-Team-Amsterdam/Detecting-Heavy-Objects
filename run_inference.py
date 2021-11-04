"""This module contains functionality to load a model and run predictions that can be
incorparated into the Azure batch processing pipeline"""
import os
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Union

import cv2
import numpy as np
import torch
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import CfgNode, get_cfg
from detectron2.data import MetadataCatalog
from detectron2.engine import DefaultPredictor
from detectron2.modeling import build_model
from detectron2.utils.visualizer import ColorMode, Visualizer
from matplotlib import pyplot
from PIL import Image

from dataset import get_container_dicts

CONTAINER_DETECTION_MODEL = None


def setup_cfg(config_file: Union[Path, str]) -> CfgNode:
    """
    Reads the model and inference settings from the config file
    """
    cfg = get_cfg()
    cfg.merge_from_file(config_file)

    cfg.freeze()
    return cfg


def init() -> None:
    """
    Initializes the trained model
    """
    global CONTAINER_DETECTION_MODEL  # pylint: disable=global-statement

    config_file = Path("configs/container_detection_inference.yaml")

    cfg = setup_cfg(config_file)

    # CONTAINER_DETECTION_MODEL = build_model(cfg)
    CONTAINER_DETECTION_MODEL = DefaultPredictor(cfg)

    # model_path = "weights/pretrained_model.pkl"  # temporary local path
    # DetectionCheckpointer(CONTAINER_DETECTION_MODEL).load(model_path)
    # CONTAINER_DETECTION_MODEL.eval()


def run(minibatch: Iterable[Union[Path, str]]) -> List[Dict[Union[Path, str], Any]]:
    """
    Processes
    Args:
        minibatch: Batch of image paths to be processed during one forward pass of
        the model
    Returns: List of dictionaries with as key unique image name and as value
    the obtained predictions
    """
    input_tensors = [
        {"image": torch.from_numpy(np.array(Image.open(path))).permute(2, 0, 1)}
        for path in minibatch
    ]

    with torch.no_grad():  # type: ignore
        outputs = CONTAINER_DETECTION_MODEL(input_tensors)  # type: ignore

    return [{path: outputs[idx]} for idx, path in enumerate(minibatch)]


def visualize_predictions() -> None:
    dataset_dicts = get_container_dicts("data_local/val")
    container_metadata = MetadataCatalog.get("container_train")

    for d in random.sample(dataset_dicts, 3):
        im = cv2.imread(d["file_name"])
        outputs = CONTAINER_DETECTION_MODEL(im)
        v = Visualizer(
            im[:, :, ::-1],
            metadata=container_metadata,
            scale=0.5,
            instance_mode=ColorMode.IMAGE_BW
            # remove the colors of unsegmented pixels. This option is only available for segmentation models
        )
        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        pyplot.imshow(out.get_image()[:, :, ::-1])
        pyplot.show()


if __name__ == "__main__":
    init()
    visualize_predictions()
