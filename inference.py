"""This module contains functionality to load a model and run predictions that can be
incorporated into the Azure batch processing pipeline"""
import json
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Union

import cv2
import numpy as np
import torch
from azureml.core import Model, Workspace
from detectron2.config import CfgNode, get_cfg
from detectron2.data import MetadataCatalog, build_detection_test_loader
from detectron2.engine import DefaultPredictor
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.utils.visualizer import ColorMode, Visualizer
from matplotlib import pyplot
from PIL import Image

from dataset import DATASET_NAME, register_dataset
from evaluation import CustomCOCOEvaluator  # type:ignore
from utils import get_container_dicts
from configs.config_parser import arg_parser

CONTAINER_DETECTION_MODEL = None
MODEL_NAME = "dummy_detectron"
VERSION = 2


def setup_cfg(config_file: Union[Path, str]) -> CfgNode:
    """
    Reads the model and inference settings from the config file
    """
    cfg = get_cfg()
    cfg.merge_from_file(config_file)

    cfg.freeze()
    return cfg


def init_inference() -> CfgNode:
    """
    Initializes the trained model

    Returns the configuration object
    """
    global CONTAINER_DETECTION_MODEL  # pylint: disable=global-statement

    config_file = Path("configs/container_detection_inference.yaml")

    cfg = setup_cfg(config_file)

    CONTAINER_DETECTION_MODEL = DefaultPredictor(cfg)

    return cfg


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


def plot_instance_segm(
    dataset_dicts: List[Dict[str, Any]],
    dataset_metadata: Dict[Any, Any],
    mode: str,
    n_sample: int,
) -> None:
    """
    Visualize the annotations of randomly selected samples in the dataset.
    Additionally, you can specify a trained model to display
    the confidence score for each annotation

    :param dataset_dicts: images metadata
    :param dataset_metadata: dataset metadata
    :param mode: the type of visualization, i.e. whether to view the object annotations
                or the confidence scores.
                 For the second option, there must be a trained model specified
                 in the configurations. Options: [ann, pred]
    :param n_sample: number of samples to be visualized
    :
    """

    for dataset_dict in random.sample(dataset_dicts, n_sample):
        img = cv2.imread(dataset_dict["file_name"])
        visualizer = Visualizer(
            img[:, :, ::-1],
            metadata=dataset_metadata,
            scale=0.5,
            instance_mode=ColorMode.IMAGE_BW,
        )
        out: DefaultPredictor = None
        if mode == "pred":
            outputs = CONTAINER_DETECTION_MODEL(img)  # type: ignore
            out = visualizer.draw_instance_predictions(outputs["instances"].to("cpu"))
        if mode == "ann":
            out = visualizer.draw_dataset_dict(dataset_dict)
        pyplot.imshow(out.get_image()[:, :, ::-1])
        pyplot.show()


def visualize_predictions(model_name: str, version: int) -> None:
    """
    This method takes a trained model from Azure, downloads it locally and plots
    visualization of randomly selected images from the validation folder

    :param model_name: name of the trained model
    :param version: version of the trained model
    """
    # TODO: make sure the correct model is downloaded from Azure.
    # download model from Azure
    ws = Workspace.from_config()
    model = Model(ws, model_name, version=version)
    # _ = model.download(target_dir="output", exist_ok=True)

    _ = init_inference()
    container_dicts = get_container_dicts("data/train")
    metadata = MetadataCatalog.get(f"{DATASET_NAME}_train")
    plot_instance_segm(container_dicts, metadata, mode="pred", n_sample=10)


def evaluate_model() -> None:
    """
    This method calculates evaluation metrics for the trained model.
    """
    # pylint: disable=too-many-function-args
    # note to self: pylint wrongly detects that build_detection_test_loader is called with too many positional args
    #  but the method has a decorator with which we can use the constructor from another method
    # i.e. _test_loader_from_config
    register_dataset(name=DATASET_NAME)
    cfg = init_inference()
    CONTAINER_DETECTION_MODEL = DefaultPredictor(cfg)

    evaluator = CustomCOCOEvaluator(
        f"{DATASET_NAME}_val", output_dir="output"
    )  # we evaluate on the validation set
    val_loader = build_detection_test_loader(cfg, f"{DATASET_NAME}_val", mapper=None)
    print(inference_on_dataset(CONTAINER_DETECTION_MODEL.model, val_loader, evaluator))


if __name__ == "__main__":
    flags = arg_parser()
    #evaluate_model()
    visualize_predictions(flags.name, flags.version)
