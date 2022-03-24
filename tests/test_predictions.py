import os
import shutil
import random
from pathlib import Path

import cv2
from detectron2.data import MetadataCatalog
from detectron2.utils.visualizer import Visualizer, ColorMode

from utils import ExperimentConfig, register_dataset, get_container_dicts


def test_visualize_predictions():

    # sanity check for upscaled images

    experimentConfig = ExperimentConfig(dataset_name="container-subset-upscaled-images",
                                        subset="train",
                                        data_format="coco")

    register_dataset(name=experimentConfig.dataset_name, data_format=experimentConfig.data_format, data_folder="data_resized")
    container_dicts = get_container_dicts(experimentConfig, data_folder="data_resized")
    metadata = MetadataCatalog.get(f"{experimentConfig.dataset_name}_{experimentConfig.subset}")

    temp_output_dir = "temp"
    os.mkdir(temp_output_dir)

    available_panos = Path("data_resized", "train").glob('*.jpg')
    upscaled_subset_annotations = []

    pano_filenames = [pano.parts[-1] for pano in available_panos]

    for image in container_dicts:
        filename = image["file_name"].split("/")[-1]
        for pano_filename in pano_filenames:
            if pano_filename == filename:
                upscaled_subset_annotations.append(image)
                break

    for i, dataset_dict in enumerate(random.sample(upscaled_subset_annotations, 10)):
        img = cv2.imread(dataset_dict["file_name"])
        visualizer = Visualizer(
            img[:, :, ::-1],
            metadata=metadata,
            scale=1,
            instance_mode=ColorMode.IMAGE,
        )

        out = visualizer.draw_dataset_dict(dataset_dict)
        cv2.imwrite(f"{temp_output_dir}/{i}.jpg", out.get_image()[:, :, ::-1])

    os.system(f"labelImg {temp_output_dir}")
    shutil.rmtree(temp_output_dir)
