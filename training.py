"""
This module contains functionality to train a default detectron2 model.
"""

import os
from datetime import datetime
from pathlib import Path
from detectron2.engine import DefaultTrainer

from configs.config_parser import arg_parser
from utils import register_dataset
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

    cfg.OUTPUT_DIR = f"{cfg.OUTPUT_DIR}/TRAIN_{flags.name}_{flags.version}"

    Path(cfg.OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    CONTAINER_DETECTION_MODEL = DefaultTrainer(cfg)
    CONTAINER_DETECTION_MODEL.resume_or_load(resume=True)
    CONTAINER_DETECTION_MODEL.train()


if __name__ == "__main__":

    flags = arg_parser()

    register_dataset(name=flags.dataset_name, data_format="coco", data_folder="data")
    print(f"Using {flags.config} config file.")
    init_train(flags)
