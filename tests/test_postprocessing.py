from unittest import TestCase
import postprocessing

class Test(TestCase):

    def __init__(self, *args, **kwargs):
        super(Test, self).__init__(*args, **kwargs)
        self.results_file = "/Users/dianaepureanu/Documents/Projects/Detecting-Heavy-Objects/outputs/" \
                        "INFER_2kx4k_resolution_1_Mar-27-01:43/coco_instances_results.json"

    def test_discard_objects(self):
        large_predictions = postprocessing.discard_objects(self.results_file, smaller_than=20000)
