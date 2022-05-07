"""This module contains functionality to load a model and run predictions that can be
incorporated into the Azure batch processing pipeline"""

# import os
# print(os.system("ls azureml-models/detectron_28feb/2"))
import argparse
import json
import logging
import pickle
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Union

import cv2
import numpy as np
import torch
from detectron2.config import CfgNode, get_cfg
from detectron2.data import MetadataCatalog, build_detection_test_loader
from detectron2.engine import DefaultPredictor
from detectron2.evaluation import inference_on_dataset
from detectron2.utils.visualizer import Visualizer
from PIL import Image

from configs.config_parser import arg_parser
from evaluation import CustomCOCOEvaluator  # type:ignore
from utils import ExperimentConfig, register_dataset

logging.basicConfig(level=logging.INFO)


def setup_cfg(config_file: Union[Path, str]) -> CfgNode:
    """
    Reads the model and inference settings from the config file
    """
    cfg = get_cfg()
    cfg.merge_from_file(config_file)

    return cfg


def init_inference(flags: argparse.Namespace) -> CfgNode:
    """
    Initializes the trained model

    :param flags: console arguments
    Returns the configuration object
    """
    config_file = Path(flags.config)

    cfg = setup_cfg(config_file)
    version = flags.version if flags.version else 1
    cfg.MODEL.DEVICE = flags.device
    cfg.MODEL.WEIGHTS = f"azureml-models/{flags.name}/{version}/model_final.pth"

    # inference run locally
    if cfg.MODEL.DEVICE == "cpu":
        cfg.MODEL.WEIGHTS = (
            f"{cfg.OUTPUT_DIR}/TRAIN_{flags.name}_{version}/model_final.pth"
        )

    return cfg


# TODO check why this function has no usages
def run(
    predictor: DefaultPredictor, minibatch: Iterable[Union[Path, str]]
) -> List[Dict[Union[Path, str], Any]]:
    """
    Processes
    Args:
        :param minibatch: Batch of image paths to be processed during one forward pass of the model
        :param predictor: detectron predictor used at inference time
    Returns: List of dictionaries with as key unique image name and as value
    the obtained predictions
    """

    input_tensors = [
        {"image": torch.from_numpy(np.array(Image.open(path)))} for path in minibatch
    ]

    with torch.no_grad():  # type: ignore

        # called, not instantiated here
        outputs = [predictor(input_tensor["image"]) for input_tensor in input_tensors]

    return [{path: outputs[idx]} for idx, path in enumerate(minibatch)]


def evaluate_model(flags: argparse.Namespace, expCfg: ExperimentConfig) -> None:
    """
    This method calculates evaluation metrics for the trained model.
    :param expCfg: experiment we evaluate
    """
    # pylint: disable=too-many-function-args
    """
    note to self: pylint wrongly detects that build_detection_test_loader is called with 
    too many positional args but the method has a decorator with which we can use the
    constructor from another method i.e. _test_loader_from_config
    """

    register_dataset(expCfg)
    cfg = init_inference(flags)
    predictor = DefaultPredictor(cfg)

    logging.info(f"Loaded model weights from {cfg.MODEL.WEIGHTS}.")

    run_name = datetime.now().strftime("%b-%d-%H:%M")
    version = flags.version if flags.version else 1
    output_dir = f"{cfg.OUTPUT_DIR}/INFER_{flags.name}_{version}_{run_name}"
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    evaluator = CustomCOCOEvaluator(
        f"{expCfg.dataset_name}_{expCfg.subset}",
        output_dir=output_dir,
        tasks=("bbox", "segm"),
    )
    loader = build_detection_test_loader(
        cfg, f"{expCfg.dataset_name}_{expCfg.subset}", mapper=None
    )
    print(inference_on_dataset(predictor.model, loader, evaluator))


if __name__ == "__main__":

    flags = arg_parser()
    experimentConfig = ExperimentConfig(
        dataset_name=flags.dataset_name,
        subset=flags.subset,
        data_format=flags.data_format,
        data_folder=flags.data_folder,
    )
    evaluate_model(flags, experimentConfig)