import json
from unittest import TestCase
from pathlib import Path
from postprocessing import PostProcessing
from visualizations.stats import DataStatistics


class Test(TestCase):

    def __init__(self, *args, **kwargs):
        super(Test, self).__init__(*args, **kwargs)
        self.results_file = Path("/Users/dianaepureanu/Documents/Projects/Detecting-Heavy-Objects/outputs/"
                                 "INFER_2kx4k_resolution_1_Mar-27-01:43/coco_instances_results.json")

    def test_discard_objects(self):
        postprocessing = PostProcessing(json_predictions=self.results_file)
        postprocessing.filter_by_size()

        filtered_stats = DataStatistics(json_file=postprocessing.output_path)
        for area in filtered_stats.areas:
            if area <= postprocessing.threshold:
                assert False, f"There are still containers smaller than {postprocessing.threshold} pixels" \
                              f" which have not been discarded."



