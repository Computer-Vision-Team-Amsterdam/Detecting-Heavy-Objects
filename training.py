import os
from pathlib import Path
from typing import Union

from detectron2.config import CfgNode, get_cfg
from detectron2.engine import DefaultTrainer

from dataset import register_dataset
from run_inference import setup_cfg

CONTAINER_DETECTION_MODEL = None

import sys
import argparse

print("############### CWD")

print(os.getcwd())

def init() -> None:
    """
    Loads a pre-trained model and fine-tunes it on the data_local dataset
    """
    global CONTAINER_DETECTION_MODEL  # pylint: disable=global-statement
    config_file = Path("/opt/pysetup/configs/container_detection_train.yaml")

    cfg = setup_cfg(config_file)

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    CONTAINER_DETECTION_MODEL = DefaultTrainer(cfg)
    CONTAINER_DETECTION_MODEL.resume_or_load(resume=False)
    CONTAINER_DETECTION_MODEL.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str)
    opt = parser.parse_args()
    print("2")
    #register_dataset()
    #init()
