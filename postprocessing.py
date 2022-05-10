"""
This model applies different post processing steps on the results file of the detectron2 model
"""
import json
from copy import deepcopy
from pathlib import Path

import pycocotools.mask as mask_util
import torch
from panorama import models
from panorama.client import PanoramaClient
from PIL import Image, ImageDraw
from tqdm import tqdm
from triangulation.masking import get_side_view_of_pano
from triangulation.triangulate import triangulate

from visualizations.stats import DataStatistics


class PostProcessing:
    """
    Post processing operations on the output of the predictor
    """

    def __init__(
        self,
        json_predictions: Path,
        threshold: float = 20000,
        mask_degrees: float = 90,
        output_folder: Path = Path.cwd(),
        output_file=Path("points_of_interest.csv"),
    ):
        """
        Args:
            :param json_predictions: path to ouput file with predictions from detectron2
            :param threshold: objects with a bbox smaller and equal to this arg are discarded. value is in pixels
            :param output_path: where the filtered json with predictions is stored
            :param predictions_to_keep: list with predictions of the model that we keep throughout postprocessing steps
        """
        self.stats = DataStatistics(json_file=json_predictions)
        self.non_filtered_stats = deepcopy(self.stats)
        self.threshold = threshold
        self.mask_degrees = mask_degrees  # todo: Add to args list
        self.output_folder = output_folder
        if isinstance(self.output_folder, str):
            self.output_folder = Path(self.output_folder)
        self.output_folder.mkdir(exist_ok=True, parents=True)
        self.output_file = output_file

    def find_points_of_interest(self):
        """
        Finds the points of interest given by COCO format json file, outputs a csv file
        with lon lat coordinates
        """
        triangulate(
            self.output_folder / "processed_predictions.json",
            self.output_folder / self.output_file,
        )

    def filter_by_angle(self):
        predictions_to_keep = []
        print("Filtering based on angle")
        for prediction in tqdm(self.stats.data):
            response = PanoramaClient.get_panorama(
                prediction["image_id"].rsplit(".", 1)[0]
            )  # TODO: Not query, look at the database!
            heading = response.heading
            height, width = prediction["segmentation"]["size"]
            mask = torch.from_numpy(
                get_side_view_of_pano(width, height, heading, self.mask_degrees)
            )[
                :, :, 0
            ]  # mask is 3D
            mask = (
                1 - mask.float()
            )  # Inverse mask to have region that we would like to keep
            segmentation_mask = torch.from_numpy(
                mask_util.decode(prediction["segmentation"])
            ).float()
            overlap = segmentation_mask * mask
            # prediction fully in side view
            if torch.sum(overlap) == torch.sum(segmentation_mask):
                predictions_to_keep.append(prediction)
            # prediction partially in side view
            # Keep predictions if minimal 50% is in the side view
            elif torch.sum(overlap) / torch.sum(segmentation_mask) > 0.5:
                predictions_to_keep.append(prediction)
        print(
            f"{len(self.stats.data) - len(predictions_to_keep)} out of the {len(self.stats.data)} are filtered based on the angle"
        )
        self.stats.update(predictions_to_keep)

    def filter_by_size(self):
        """
        Removes predictions of small objects and writes results to json file
        """
        indices_to_keep = [
            idx for idx, area in enumerate(self.stats.areas) if area > self.threshold
        ]
        self.stats.update([self.stats.data[idx] for idx in indices_to_keep])

    def save_filtered_predictions(self, name):
        with open(self.output_folder / "processed_predictions.json", "w") as f:
            json.dump(self.stats.data, f)


if __name__ == "__main__":
    postprocess = PostProcessing("coco_instances_results.json")
    postprocess.filter_by_size()
    postprocess.filter_by_angle()
    postprocess.save_filtered_predictions("filtered_predictions.json")
    postprocess.find_points_of_interest()
