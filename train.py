import os
from pathlib import Path
from typing import Union

from dataset import register_dataset
from detectron2.config import CfgNode, get_cfg
from detectron2.engine import DefaultTrainer

CONTAINER_DETECTION_MODEL = None


def setup_train_cfg(config_file: Union[Path, str]) -> CfgNode:
    """
    Reads the model and inference settings from the config file
    """
    cfg = get_cfg()
    cfg.merge_from_file(config_file)
    cfg.DATASETS.TRAIN = ("container_train",)  # TODO: remove hardcoded value
    cfg.DATASETS.TEST = ()
    cfg.DATALOADER.NUM_WORKERS = 1
    cfg.MODEL.WEIGHTS = "detectron2://ImageNetPretrained/MSRA/R-50.pkl"  # TODO: remove hardcoded value
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR = 0.00025
    cfg.SOLVER.MAX_ITER = 3
    cfg.SOLVER.STEPS = []  # do not decay learning rate
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128  # faster, and good enough for this toy dataset (default: 512)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only has one class (data)
    cfg.MODEL.DEVICE = "cpu" # TODO: remove hardcoded value
    cfg.freeze()

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    return cfg


def init() -> None:
    """
    Loads a pre-trained model and fine-tunes it on the data dataset
    """
    global CONTAINER_DETECTION_MODEL  # pylint: disable=global-statement

    config_file = Path("configs/container_detection.yaml")

    cfg = setup_train_cfg(config_file)

    CONTAINER_DETECTION_MODEL = DefaultTrainer(cfg)
    CONTAINER_DETECTION_MODEL.resume_or_load(resume=False)
    CONTAINER_DETECTION_MODEL.train()


if __name__ == "__main__":

    register_dataset()
    init()





