"""
This module contains functionality to train a default detectron2 model.
"""
import argparse
from pathlib import Path

from detectron2.engine import DefaultTrainer

from configs.config_parser import arg_parser
from inference import setup_cfg
from utils import ExperimentConfig, register_dataset


def init_train(flags: argparse.Namespace) -> None:
    """
    Loads a pre-trained model and fine-tunes it on the data dataset
    """
    config_file = Path(flags.config)

    cfg = setup_cfg(config_file)
    cfg.MODEL.DEVICE = flags.device

    cfg.OUTPUT_DIR = f"{cfg.OUTPUT_DIR}/TRAIN_{flags.name}_{flags.version}"

    Path(cfg.OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    model = DefaultTrainer(cfg)
    model.resume_or_load(resume=True)
    model.train()


if __name__ == "__main__":

    flags = arg_parser()
    experimentConfig = ExperimentConfig(
        dataset_name=flags.dataset_name,
        subset=flags.subset,
        data_format=flags.data_format,
        data_folder=flags.data_folder,
    )

    register_dataset(experimentConfig)
    init_train(flags)
