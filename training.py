"""
This module contains functionality to train a default detectron2 model.
"""
import argparse
from pathlib import Path

from detectron2.engine import DefaultTrainer

from configs.config_parser import arg_parser
from evaluation import CustomCOCOEvaluator
from inference import setup_cfg
from utils import ExperimentConfig, register_dataset


class MyTrainer(DefaultTrainer):

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        output_dir = f"{cfg.OUTPUT_DIR}"
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        evaluator = CustomCOCOEvaluator(
            f"container_val",
            output_dir=output_dir,
            tasks=("bbox", "segm"),
        )

        return evaluator


def init_train(flags: argparse.Namespace) -> None:
    """
    Loads a pre-trained model and fine-tunes it on the data dataset
    """
    config_file = Path(flags.config)

    cfg = setup_cfg(config_file)
    cfg.MODEL.DEVICE = flags.device

    print(f"model weights: {cfg.MODEL.WEIGHTS}")
    cfg.OUTPUT_DIR = f"{cfg.OUTPUT_DIR}/TRAIN_{flags.name}_{flags.version}"

    Path(cfg.OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    model = MyTrainer(cfg)
    model.resume_or_load(resume=True)
    model.train()


if __name__ == "__main__":

    flags = arg_parser()
    print("Flags")
    print(flags)
    experimentConfig = ExperimentConfig(
        dataset_name=flags.dataset_name,
        subset=flags.subset,
        data_format=flags.data_format,
        data_folder=flags.data_folder,
    )

    register_dataset(experimentConfig)
    init_train(flags)
