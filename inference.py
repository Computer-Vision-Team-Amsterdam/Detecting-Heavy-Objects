"""This module contains functionality to load a model and run predictions that can be
incorporated into the Azure batch processing pipeline"""

# import os
# print(os.system("ls azureml-models/detectron_28feb/2"))
import json
import pickle
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Union

import cv2
import numpy as np
import torch

from detectron2.config import CfgNode, get_cfg
from detectron2.data import build_detection_test_loader
from detectron2.engine import DefaultPredictor
from detectron2.evaluation import inference_on_dataset

from PIL import Image

from configs.config_parser import arg_parser
from utils import register_dataset, register_test_dataset
from evaluation import CustomCOCOEvaluator  # type:ignore
from utils import ExperimentConfig

import logging
logging.basicConfig(level=logging.INFO)

CONTAINER_DETECTION_MODEL = None


def setup_cfg(config_file: Union[Path, str]) -> CfgNode:
    """
    Reads the model and inference settings from the config file
    """
    cfg = get_cfg()
    cfg.merge_from_file(config_file)

    return cfg


def init_inference(flags) -> CfgNode:
    """
    Initializes the trained model

    Returns the configuration object
    """
    global CONTAINER_DETECTION_MODEL  # pylint: disable=global-statement

    config_file = Path(flags.config)

    cfg = setup_cfg(config_file)
    cfg.MODEL.DEVICE = flags.device
    cfg.MODEL.WEIGHTS = f"azureml-models/{flags.name}/{flags.version}/model_final.pth"

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
        {"image": torch.from_numpy(np.array(Image.open(path)))}
        for path in minibatch]

    with torch.no_grad():  # type: ignore

        # called, not instantiated here
        outputs = [CONTAINER_DETECTION_MODEL(input_tensor["image"]) for input_tensor in input_tensors]  # type: ignore

    return [{path: outputs[idx]} for idx, path in enumerate(minibatch)]


def evaluate_model(flags, expCfg: ExperimentConfig) -> None:
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

    #register_dataset(name=expCfg.dataset_name, data_format=expCfg.data_format, data_folder="data")
    register_test_dataset(name=expCfg.dataset_name,data_format=expCfg.data_format, data_folder="data")
    cfg = init_inference(flags)
    CONTAINER_DETECTION_MODEL = DefaultPredictor(cfg)

    logging.info(f"Loaded model weights from {cfg.MODEL.WEIGHTS}.")

    run_name = datetime.now().strftime("%b-%d-%H:%M")
    output_dir = f"{cfg.OUTPUT_DIR}/INFER_{flags.name}_{flags.version}_{run_name}"
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    evaluator = CustomCOCOEvaluator(f"{expCfg.dataset_name}_{expCfg.subset}", output_dir=output_dir, tasks=("bbox", "segm"))
    loader = build_detection_test_loader(cfg, f"{expCfg.dataset_name}_{expCfg.subset}", mapper=None)
    print(inference_on_dataset(CONTAINER_DETECTION_MODEL.model, loader, evaluator))


if __name__ == "__main__":

    """
    flags = arg_parser()

    experimentConfig = ExperimentConfig(dataset_name=flags.dataset_name,
                                        subset=flags.subset,
                                        data_format=flags.data_format)
    evaluate_model(flags, experimentConfig)
    """

    flags = arg_parser()
    experimentConfig = ExperimentConfig(dataset_name=flags.dataset_name,
                                        subset="test",
                                        data_format=flags.data_format)
    evaluate_model(flags, experimentConfig)


    """
    flags = arg_parser()
    cfg = init_inference(flags)
    CONTAINER_DETECTION_MODEL = DefaultPredictor(cfg)

    images = Path("data_azure/17mar2021/0/blurred").glob("*.jpg")
    count = 0
    results = []
    for image in images:
        count = count + 1
        if count == 2:
            break
        im = cv2.imread(str(image))
        out = CONTAINER_DETECTION_MODEL(im)
        out["image_id"] = image.stem
        results.append(out)
        print(results)

    with open("map_info.pkl", "w") as f :
        pickle.dump(results, f)
    
    """