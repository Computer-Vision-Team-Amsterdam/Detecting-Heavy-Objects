"""This module contains functionality to load a model and run predictions that can be
incorparated into the Azure batch processing pipeline"""

from pathlib import Path
from typing import Any, Dict, Iterable, List, Union

import numpy as np
import torch
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import CfgNode, get_cfg
from detectron2.modeling import build_model
from PIL import Image

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

    config_file = Path("configs/container_detection.yaml")

    cfg = setup_cfg(config_file)

    CONTAINER_DETECTION_MODEL = build_model(cfg)

    model_path = "weights/pretrained_model.pkl"  # temporary local path
    DetectionCheckpointer(CONTAINER_DETECTION_MODEL).load(model_path)
    CONTAINER_DETECTION_MODEL.eval()


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
