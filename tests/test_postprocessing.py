import json
from pathlib import Path
from typing import Any
from unittest import TestCase

from postprocessing import PostProcessing
from visualizations.stats import DataStatistics


class Test(TestCase):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super(Test, self).__init__(*args, **kwargs)
        script_location = Path(__file__).absolute().parent
        self.results_file = Path(script_location, "coco_instances_results.json")

    def test_discard_objects(self) -> None:
        postprocessing = PostProcessing(json_predictions=self.results_file)
        postprocessing.filter_by_size()

        filtered_stats = DataStatistics(json_file=postprocessing.output_path)
        for area in filtered_stats.areas:
            assert area > postprocessing.threshold, (
                f"There are still containers smaller than "
                f"{postprocessing.threshold} pixels"
                f" which have not been discarded."
            )
