"""
This model applies different post processing steps on the results file of the detectron2 model
"""
import json
from pathlib import Path
from typing import List

from visualizations.stats import DataStatistics


class PostProcessing:
    """
    Post processing operations on the output of the predictor
    """

    def __init__(
        self,
        json_predictions: Path,
        threshold: float = 20000,
        output_path: Path = Path.cwd() / "postprocessed.json",
    ) -> None:
        """
        Args:
            :param json_predictions: path to ouput file with predictions from detectron2
            :param threshold: objects with a bbox smaller and equal to this arg are discarded. value is in pixels
            :param output_path: where the filtered json with predictions is stored
        """
        self.stats = DataStatistics(json_file=json_predictions)
        self.threshold = threshold
        self.output_path = output_path
        self.predictions_to_keep: List[str] = []

    def filter_by_size(self) -> None:
        """
        Removes predictions of small objects and writes results to json file
        """

        def remove_predictions() -> None:
            """
            Filter out all predictions where area of the bbox of object is smaller and equal to @param threshold pixels
            """
            indices_to_keep = [
                idx
                for idx, area in enumerate(self.stats.areas)
                if area > self.threshold
            ]
            self.predictions_to_keep = [self.stats.data[idx] for idx in indices_to_keep]

        def write_json() -> None:
            """
            Write filtered list of predictions to another json file
            """
            with open(self.output_path, "w") as f:
                json.dump(self.predictions_to_keep, f)
            f.close()

        remove_predictions()
        write_json()
        print(self.output_path)
