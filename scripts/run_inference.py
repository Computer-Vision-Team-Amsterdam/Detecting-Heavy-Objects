"""This module contains functionality to load a model and run predictions that can be
incorparated into the Azure batch processing pipeline"""
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Union

import numpy as np
import torch
from azureml.core import Model
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

    import subprocess
    print(subprocess.run(["ls", "-la"]))
    config_file = Path("configs/container_detection.yaml")
    print(config_file)

    cfg = setup_cfg(config_file)
    print(cfg)

    CONTAINER_DETECTION_MODEL = build_model(cfg)
    print(CONTAINER_DETECTION_MODEL)

    # model_path = os.getenv("AZURE_MODEL_DIR")  # temporary local path
    # model_path = "weights/pretrained_model.pkl"  # temporary local path
    model_path = Model.get_model_path('classification_model')
    print(model_path)
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


# if __name__ == "__main__":
#     import sys
#     init()
#
#     import os
#     print("AZUREML_MODEL_DIR", os.environ.get("AZUREML_MODEL_DIR"))
#
#     with open(sys.argv[1], "rb") as fh:
#         content = fh.read()
#         print(content[:100])
#
#     print(run([sys.argv[1]]))
