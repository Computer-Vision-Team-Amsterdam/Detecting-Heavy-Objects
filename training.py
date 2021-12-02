"""
This module contains functionality to train a default detectron2 model.
"""

import argparse
import os
from pathlib import Path

from detectron2.engine import DefaultTrainer

from dataset import DATASET_NAME, register_dataset
from inference import setup_cfg

CONTAINER_DETECTION_MODEL = None


def init() -> None:
    """
    Loads a pre-trained model and fine-tunes it on the data dataset
    """
    global CONTAINER_DETECTION_MODEL  # pylint: disable=global-statement
    config_file = Path("configs/container_detection_train.yaml")

    cfg = setup_cfg(config_file)
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    CONTAINER_DETECTION_MODEL = DefaultTrainer(cfg)
    CONTAINER_DETECTION_MODEL.resume_or_load(resume=False)
    CONTAINER_DETECTION_MODEL.train()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset")
    opt = parser.parse_args()
    register_dataset(DATASET_NAME)
    init()
