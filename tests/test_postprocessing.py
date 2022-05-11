from pathlib import Path
from typing import Any
from unittest import TestCase

from postprocessing import PostProcessing


class Test(TestCase):
    @pytest.mark.skip(
        reason="temporarily skipping this because it requires GDAL"
    )
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super(Test, self).__init__(*args, **kwargs)
        script_location = Path(__file__).absolute().parent
        self.results_file = Path(script_location, "coco_instances_results.json")

    def test_discard_objects(self) -> None:
        postprocessing = PostProcessing(json_predictions=self.results_file)
        postprocessing.filter_by_size()

        for area in postprocessing.stats.areas:
            assert area > postprocessing.threshold, (
                f"There are still containers smaller than "
                f"{postprocessing.threshold} pixels"
                f" which have not been discarded."
            )
