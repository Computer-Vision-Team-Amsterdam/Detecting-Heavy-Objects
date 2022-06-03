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

import albumentations as A
from detectron2.structures import BoxMode

import numpy as np
from detectron2.data.transforms import Augmentation
from fvcore.transforms.transform import Transform, NoOpTransform

class AlbumentationsTransform(Transform):
    def __init__(self, aug, params):
        self.aug = aug
        self.params = params

    def apply_coords(self, coords: np.ndarray) -> np.ndarray:
        return coords

    def apply_image(self, image):
        return self.aug.apply(image, **self.params)

    def apply_box(self, box: np.ndarray) -> np.ndarray:
        try:
            return np.array(self.aug.apply_to_bboxes(box.tolist(), **self.params))
        except AttributeError:
            return box

    def apply_segmentation(self, segmentation: np.ndarray) -> np.ndarray:
        try:
            return self.aug.apply_to_mask(segmentation, **self.params)
        except AttributeError:
            return segmentation


class AlbumentationsWrapper(Augmentation):
    """
    Wrap an augmentor form the albumentations library: https://github.com/albu/albumentations.
    Image, Bounding Box and Segmentation are supported.
    """

    def __init__(self, augmentor):
        """
        Args:
            augmentor (albumentations.BasicTransform):
        """
        # super(Albumentations, self).__init__() - using python > 3.7 no need to call rng
        self._aug = augmentor

    def get_transform(self, image):
        do = self._rand_range() < self._aug.p
        if do:
            params = self.prepare_param(image)
            return AlbumentationsTransform(self._aug, params)
        else:
            return NoOpTransform()

    def prepare_param(self, image):
        params = self._aug.get_params()
        if self._aug.targets_as_params:
            targets_as_params = {"image": image}
            params_dependent_on_targets = self._aug.get_params_dependent_on_targets(targets_as_params)
            params.update(params_dependent_on_targets)
        params = self._aug.update_params(params, **{"image": image})
        return params


#================= END AUG WRAPPER ================ #

def custom_mapper_wrapper(dataset_dict):
    dataset_dict = copy.deepcopy(dataset_dict) 
    image = detection_utils.read_image(dataset_dict["file_name"], format="RGB")

    # ========== START ALBUMENTATIONS =========== #

    augs = T.AugmentationList([
        AlbumentationsWrapper(A.GaussNoise(p=0.2)),
        AlbumentationsWrapper(A.MedianBlur(blur_limit=3, p=0.1)),
        AlbumentationsWrapper(A.CLAHE(clip_limit=2)),
        AlbumentationsWrapper(A.RandomBrightnessContrast()),
        AlbumentationsWrapper(A.HueSaturationValue(p=0.3))

    ])
    input = T.AugInput(image)
    _ = augs(input)
    image = input.image  # new image

    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # ========== END ALBUMENTATIONS =========== #

    # ========== START DET2 AUGMENTATIONS =========== #

    transform_list = [T.RandomFlip(prob=0.5, horizontal=True, vertical=False),
                      T.RandomRotation(angle=[-30, 30], expand=False),
                      T.RandomCrop(crop_type="relative_range", crop_size=(0.7, 0.7))]

    image, transforms = T.apply_transform_gens(transform_list, image)
    dataset_dict["image"] = torch.as_tensor(image.transpose((2, 0, 1)).astype("float32"))
    annos = [
        detection_utils.transform_instance_annotations(obj, transforms, image.shape[:2])
        for obj in dataset_dict.pop("annotations")
        if obj.get("iscrowd", 0) == 0
    ]

    # ========== END DET2 AUGMENTATIONS =========== #
    instances = detection_utils.annotations_to_instances(annos, image.shape[:2])
    dataset_dict["instances"] = detection_utils.filter_empty_instances(instances)
    return dataset_dict


def augment_data(model, cfg):
    train_data_loader = model.build_train_loader(cfg)
    data_iter = iter(train_data_loader)
    batch = next(data_iter)

    rows, cols = 3, 3
    plt.figure(figsize=(20, 20))

    for i, per_image in enumerate(batch[:int(rows * cols)]):
        plt.subplot(rows, cols, i + 1)

        # Pytorch tensor is in (C, H, W) format

        img = per_image["image"].permute(1, 2, 0).cpu().detach().numpy()
        b, g, r = cv2.split(img)
        img = cv2.merge([r, g, b])
        img = detection_utils.convert_image_to_rgb(img, cfg.INPUT.FORMAT)

        metadata = MetadataCatalog.get(f"container_train")
        visualizer = Visualizer(img, metadata=metadata, scale=2)

        target_fields = per_image["instances"].get_fields()

        labels = None
        vis = visualizer.overlay_instances(
            labels=labels,
            boxes=target_fields.get("gt_boxes", None),
            masks=target_fields.get("gt_masks", None),
            keypoints=target_fields.get("gt_keypoints", None),
        )

        plt.imshow(vis.get_image()[:, :, ::-1])
    plt.show()


class MyTrainer(DefaultTrainer):

    @classmethod
    def build_train_loader(cls, cfg):
        cfg.MODEL.DEVICE = "cpu"
        cfg.SOLVER.IMS_PER_BATCH = 16
        return build_detection_train_loader(cfg, mapper=custom_mapper_wrapper)

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

    augment_data(model, cfg)

    #model.resume_or_load(resume=True)
    #model.train()


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
