"""
Visualize predictions or annotations on a data subset.
"""
import argparse
import os
import random
import numpy as np
import shutil
from pathlib import Path
from typing import Any, Dict, List
from PIL import Image

import cv2
import torch
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.evaluation import inference_on_dataset
from detectron2.modeling import build_model

from azureml.core import Model, Workspace
from detectron2.data import MetadataCatalog, build_detection_test_loader
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import ColorMode, Visualizer
from tqdm import tqdm

from configs import config_parser
from evaluation import CustomCOCOEvaluator
from inference import init_inference
from utils import ExperimentConfig, get_container_dicts, register_dataset


def plot_instance_segm(
    dataset_dicts: List[Dict[str, Any]],
    dataset_metadata: Dict[Any, Any],
    mode: str,
    n_sample: int,
) -> None:
    """
    Visualize the annotations of randomly selected samples in the dataset.
    Additionally, you can specify a trained model to display
    the confidence score for each annotation
    :param dataset_dicts: images metadata
    :param dataset_metadata: dataset metadata
    :param mode: the type of visualization, i.e. whether to view the object annotations
                or the confidence scores.
                 For the second option, there must be a trained model specified
                 in the configurations. Options: [ann, pred]
    :param n_sample: number of samples to be visualized
    :
    """

    if mode == "pred":
        cfg = init_inference(flags)
        predictor = DefaultPredictor(cfg)

    temp_output_dir = "temp"
    os.mkdir(temp_output_dir)
    for i, dataset_dict in tqdm(
        enumerate(random.sample(dataset_dicts, n_sample)), total=n_sample
    ):
        img = cv2.imread(dataset_dict["file_name"])
        visualizer = Visualizer(
            img[:, :, ::-1],
            metadata=dataset_metadata,
            scale=1,
            instance_mode=ColorMode.IMAGE,
        )
        if mode == "pred":
            outputs = predictor(img)
            out = visualizer.draw_instance_predictions(outputs["instances"].to("cpu"))
        if mode == "ann":
            out = visualizer.draw_dataset_dict(dataset_dict)

        cv2.imwrite(f"{temp_output_dir}/{i}.jpg", out.get_image()[:, :, ::-1])

        """
        cv2.namedWindow('pred', cv2.WINDOW_NORMAL)
        cv2.setWindowProperty('pred', cv2.WND_PROP_TOPMOST, 1)
        cv2.imshow("pred", mat=out.get_image()[:, :, ::-1])
        cv2.waitKey(5)
        cv2.destroyAllWindows()
        """

    os.system(f"labelImg {temp_output_dir}")
    shutil.rmtree(temp_output_dir)


def visualize_predictions(flags: argparse.Namespace, expCfg: ExperimentConfig) -> None:
    """
    This method takes a trained model from Azure, downloads it locally and plots
    visualization of randomly selected images from the validation folder
    :param flags: console arguments
    :param expCfg: experiment configuration
    """

    register_dataset(expCfg)
    container_dicts = get_container_dicts(expCfg)

    metadata = MetadataCatalog.get(f"{expCfg.dataset_name}_{expCfg.subset}")
    plot_instance_segm(
        container_dicts, metadata, mode=flags.mode, n_sample=flags.n_sample
    )

def single_instance_prediction(flags: argparse.Namespace, expCfg: ExperimentConfig, image_path: Path):
    register_dataset(expCfg)

    #im = cv2.imread(image_path)
    images_paths = [image_path]


    input_tensors = [
        {"image": torch.from_numpy(np.array(Image.open(path))).permute(2, 0, 1)}
        for path in images_paths
    ]


    ws = Workspace.from_config()

    _ = Model.get_model_path(
        model_name=f"{flags.name}", version=int(flags.version), _workspace=ws
    )

    cfg = init_inference(flags)
    model = build_model(cfg)  # returns a torch.nn.Module
    DetectionCheckpointer(model).load(cfg.MODEL.WEIGHTS)
    model.eval()

    with torch.no_grad():  # type: ignore
        outputs = model(input_tensors)

    evaluator = CustomCOCOEvaluator(
        f"{expCfg.dataset_name}_{expCfg.subset}",
        output_dir="OUT",
        tasks=("bbox", "segm"),
    )
    loader = build_detection_test_loader(cfg, mapper=None)
    print(inference_on_dataset(model, loader, evaluator))
    # return [{path: outputs[idx]} for idx, path in enumerate(images_paths)]


def single_instance_prediction_2(
    flags: argparse.Namespace, expCfg: ExperimentConfig, image_path: Path
) -> None:
    register_dataset(expCfg)
    im = cv2.imread(image_path)

    ws = Workspace.from_config()

    _ = Model.get_model_path(
        model_name=f"{flags.name}", version=int(flags.version), _workspace=ws
    )

    cfg = init_inference(flags)
    predictor = DefaultPredictor(cfg)
    outputs = predictor(im)

    print(outputs["instances"])

    metadata = MetadataCatalog.get(f"{expCfg.dataset_name}_{expCfg.subset}")
    v = Visualizer(im[:, :, ::-1], metadata, scale=1.2)
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))

    # save image in current directory
    image_name = Path(image_path).stem
    cv2.imwrite(f"{image_name}:out.jpg", out.get_image()[:, :, ::-1])


if __name__ == "__main__":
    flags = config_parser.arg_parser()
    flags.device = "cpu"
    experimentConfig = ExperimentConfig(
        dataset_name=flags.dataset_name,
        subset=flags.subset,
        data_format=flags.data_format,
        data_folder=flags.data_folder,
    )

    if flags.image:
        # SINGLE IMAGE PREDICTION
        single_instance_prediction(flags, experimentConfig, flags.image)
    if not flags.image:
        visualize_predictions(flags, experimentConfig)

