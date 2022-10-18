"""
Visualize predictions or annotations on a data subset.

TODO: fix the problem below so that the code runs well from the first time.
first run python predictions.py --name best --version 1 --device cpu --image data_sample/train/pano_0000_002120.jpg
to download the model locally
then run
python predictions.py --name best --version 1 --device cpu --image data_sample/train/pano_0000_002120.jpg
again
"""
import argparse
import os
import random
import shutil
import time
from pathlib import Path
from typing import Any, Dict, List

import cv2
from azureml.core import Model, Workspace
from detectron2.data import MetadataCatalog
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import ColorMode, Visualizer
from tqdm import tqdm

from configs import config_parser
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

    print(f"leen {len(dataset_dicts)}")
    print(dataset_dicts)
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


def single_instance_prediction(
    flags: argparse.Namespace, expCfg: ExperimentConfig, image_path: Path
) -> None:
    register_dataset(expCfg)
    im = cv2.imread(image_path)

    ws = Workspace.from_config()

    azure_path_prefix = Model.get_model_path(
        model_name=f"{flags.name}", version=int(flags.version), _workspace=ws
    )

    weights_path = os.path.join(azure_path_prefix, "outputs", "model_final.pth")
    flags.weights = weights_path
    print(f"weights {flags.weights}")

    cfg = init_inference(flags)
    predictor = DefaultPredictor(cfg)
    outputs = predictor(im)

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
