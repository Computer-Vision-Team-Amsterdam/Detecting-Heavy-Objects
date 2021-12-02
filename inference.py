"""This module contains functionality to load a model and run predictions that can be
incorparated into the Azure batch processing pipeline"""

import random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Union

import cv2
import joblib as joblib
import numpy as np
import torch
from azureml.core import Model, Run, Workspace
from detectron2.config import CfgNode, get_cfg
from detectron2.data import MetadataCatalog
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import ColorMode, Visualizer
from matplotlib import pyplot
from PIL import Image

from utils import get_container_dicts

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

    CONTAINER_DETECTION_MODEL = DefaultPredictor(cfg)


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
            outputs = CONTAINER_DETECTION_MODEL(img)  # type: ignore
            out = visualizer.draw_instance_predictions(outputs["instances"].to("cpu"))
        if mode == "ann":
            out = visualizer.draw_dataset_dict(d)
        pyplot.imshow(out.get_image()[:, :, ::-1])
        pyplot.show()


if __name__ == "__main__":

    model_name = "dummy_detectron"
    version = 2

    # download model from Azure
    ws = Workspace.from_config()
    model = Model(ws, model_name, version=version)
    _ = model.download(target_dir="output", exist_ok=True)

    init()
    dataset_dicts = get_container_dicts("data/val")
    metadata = MetadataCatalog.get("container_train")
    visualize_images(dataset_dicts, metadata, mode="pred", n_sample=3)
