"""
This module contains functionality to train a default detectron2 model.
"""
import argparse
import copy
import os
import matplotlib.pyplot as plt
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetMapper

import cv2
import torch
from pathlib import Path

from detectron2.data import build_detection_train_loader
from detectron2.engine import DefaultTrainer
import detectron2.data.transforms as T
from detectron2.data import detection_utils

from configs.config_parser import arg_parser
from evaluation import CustomCOCOEvaluator
from inference import setup_cfg
from utils import ExperimentConfig, register_dataset

def custom_mapper(dataset_dict):
    # Implement a mapper, similar to the default DatasetMapper, but with your own customizations
    dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
    image = detection_utils.read_image(dataset_dict["file_name"], format="BGR")
    transform_list = [T.Resize((800, 1600)),
                      T.RandomFlip(prob=0.5, horizontal=False, vertical=True),
                      T.RandomFlip(prob=0.5, horizontal=True, vertical=False),
                      T.RandomBrightness(intensity_min=0.5, intensity_max=1.5),
                      T.RandomRotation(angle=[0, 45])]
    image, transforms = T.apply_transform_gens(transform_list, image)
    dataset_dict["image"] = torch.as_tensor(image.transpose(2, 0, 1).astype("float32"))

    annos = [
        detection_utils.transform_instance_annotations(obj, transforms, image.shape[:2])
        for obj in dataset_dict.pop("annotations")
        if obj.get("iscrowd", 0) == 0
    ]
    instances = detection_utils.annotations_to_instances(annos, image.shape[:2])
    dataset_dict["instances"] = detection_utils.filter_empty_instances(instances)
    return dataset_dict


class MyTrainer(DefaultTrainer):

    @classmethod
    def build_train_loader(cls, cfg):
        return build_detection_train_loader(cfg, mapper=custom_mapper)

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

    version = flags.version if flags.version else 1
    cfg.OUTPUT_DIR = f"{cfg.OUTPUT_DIR}/TRAIN_{flags.name}_{version}"

    Path(cfg.OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    model = MyTrainer(cfg)

    model.resume_or_load(resume=True)
    model.train()


if __name__ == "__main__":
    flags = arg_parser()
    experimentConfig_train = ExperimentConfig(
        dataset_name=flags.dataset_name,
        subset=flags.subset,
        data_format=flags.data_format,
        data_folder=flags.data_folder,
    )

    register_dataset(experimentConfig_train)

    experimentConfig_val = ExperimentConfig(
        dataset_name=flags.dataset_name,
        subset="val",
        data_format=flags.data_format,
        data_folder=flags.data_folder,
    )

    register_dataset(experimentConfig_val)
    init_train(flags)
