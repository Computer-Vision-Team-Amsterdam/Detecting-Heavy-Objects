"""
This module contains functionality to train a default detectron2 model.
"""

import argparse
import itertools
import os
import yaml
from pathlib import Path
from tqdm import trange
from detectron2.engine import DefaultTrainer

from dataset import DATASET_NAME, register_dataset
from inference import setup_cfg

CONTAINER_DETECTION_MODEL = None


def init_train(flags) -> None:
    """
    Loads a pre-trained model and fine-tunes it on the data dataset
    """
    global CONTAINER_DETECTION_MODEL  # pylint: disable=global-statement
    config_file = Path(flags.config)

    cfg = setup_cfg(config_file)
    cfg.MODEL.DEVICE = flags.device
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    CONTAINER_DETECTION_MODEL = DefaultTrainer(cfg)
    CONTAINER_DETECTION_MODEL.resume_or_load(resume=False)
    CONTAINER_DETECTION_MODEL.train()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="data/")
    parser.add_argument("--config", type=str, default="configs/temp.yaml")
    parser.add_argument("--device", type=str, default="cuda:0")

    flags = parser.parse_args()

    register_dataset(DATASET_NAME)
    print(f"Using {flags.config} config file.")
    init_train(flags)
