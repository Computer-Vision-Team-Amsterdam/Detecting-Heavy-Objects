"""
Visualize predictions or annotations on a data subset.
"""
import os
import shutil
import random
import cv2
from typing import List, Any, Dict

from azureml.core import Workspace, Model
from detectron2.data import MetadataCatalog
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer, ColorMode

from configs.config_parser import arg_parser
from dataset import register_dataset
from inference import init_inference
from utils import ExperimentConfig, get_container_dicts

CONTAINER_DETECTION_MODEL = None


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
    temp_output_dir = "temp"
    os.mkdir(temp_output_dir)
    for i, dataset_dict in enumerate(random.sample(dataset_dicts, n_sample)):
        img = cv2.imread(dataset_dict["file_name"])
        visualizer = Visualizer(
            img[:, :, ::-1],
            metadata=dataset_metadata,
            scale=1,
            instance_mode=ColorMode.IMAGE,
        )
        out: DefaultPredictor = None
        if mode == "pred":
            outputs = CONTAINER_DETECTION_MODEL(img)  # type: ignore
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


def visualize_predictions(flags, expCfg: ExperimentConfig) -> None:
    """
    This method takes a trained model from Azure, downloads it locally and plots
    visualization of randomly selected images from the validation folder

    :param expCfg: experiment configuration
    """

    register_dataset(name=expCfg.dataset_name, data_format=expCfg.data_format)
    container_dicts = get_container_dicts(expCfg)
    cfg = init_inference(flags)

    global CONTAINER_DETECTION_MODEL
    CONTAINER_DETECTION_MODEL = DefaultPredictor(cfg)

    metadata = MetadataCatalog.get(f"{expCfg.dataset_name}_{expCfg.subset}")
    plot_instance_segm(container_dicts, metadata, mode="pred", n_sample=10)


if __name__ == "__main__":
    flags = arg_parser()

    experimentConfig = ExperimentConfig(dataset_name=flags.dataset_name,
                                        subset=flags.subset,
                                        data_format=flags.data_format)
    visualize_predictions(flags, experimentConfig)

