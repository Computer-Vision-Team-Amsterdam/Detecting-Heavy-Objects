import json
import shutil
from pathlib import Path
from typing import List
from unittest.mock import Mock, call

from visualizations import get_obj_size
from utils import DataFormatConverter

from unittest import TestCase


class Test(TestCase):

    def __init__(self, *args, **kwargs):
        super(Test, self).__init__(*args, **kwargs)
        input = "/Users/dianaepureanu/Documents/Projects/containers-annotated-1151-4000x2000.json"
        output_dir = "/Users/dianaepureanu/Documents/Projects/Detecting-Heavy-objects/tests/converter_output"
        self.converter = DataFormatConverter(input, output_dir)

    def test__add_key(self):

        # firstly, discard all images with bad resolution
        # make sure we resize the 2000x1000 images to 40000x2000, then re-create the input file for this test.
        self.converter._add_key("iscrowd", 0)
        for ann in self.converter._input["annotations"]:
            self.assertTrue(("iscrowd", 0) in ann.items())

    def test__calculate_area(self):
        self.converter._to_absolute()
        self.converter._calculate_area()
        for ann in self.converter._input["annotations"]:
            self.assertTrue(("area", 0) not in ann.items())

        """
        get_obj_size.plot_avg_bbox(data=self.converter._input,
                                   output_dir="/Users/dianaepureanu/Documents/Projects/Detecting-Heavy-objects/tests/")
        """

    def test__to_absolute(self):

        relative_segm: List[List[float]] = [ann["segmentation"][0] for ann in self.converter._input["annotations"]]
        relative_bbox: List[List[float]] = [ann["bbox"] for ann in self.converter._input["annotations"]]

        self.converter._to_absolute()

        absolute_segm: List[List[float]] = [ann["segmentation"][0] for ann in self.converter._input["annotations"]]
        absolute_bbox: List[List[float]] = [ann["bbox"] for ann in self.converter._input["annotations"]]

        for rel_segm, abs_segm in zip(relative_segm, absolute_segm):
            rel_xs = [rel_segm[i] * 4000 for i in range(len(rel_segm)) if i % 2 == 0]
            rel_ys = [rel_segm[i] * 2000 for i in range(len(rel_segm)) if i % 2 == 1]

            abs_xs = [abs_segm[i] for i in range(len(abs_segm)) if i % 2 == 0]
            abs_ys = [abs_segm[i] for i in range(len(abs_segm)) if i % 2 == 1]

            self.assertListEqual(rel_xs, abs_xs)
            self.assertListEqual(rel_ys, abs_ys)

        for rel_bbox, abs_bbox in zip(relative_bbox, absolute_bbox):
            rel_xs = [rel_bbox[i] * 4000 for i in range(len(rel_bbox)) if i % 2 == 0]
            rel_ys = [rel_bbox[i] * 2000 for i in range(len(rel_bbox)) if i % 2 == 1]

            abs_xs = [abs_bbox[i] for i in range(len(abs_bbox)) if i % 2 == 0]
            abs_ys = [abs_bbox[i] for i in range(len(abs_bbox)) if i % 2 == 1]

            self.assertListEqual(rel_xs, abs_xs)
            self.assertListEqual(rel_ys, abs_ys)

    def test__split(self):

        self.converter._split()

        subset_count = 0
        for subset in ["train", "val", "test"]:
            output_name = f"{self.converter._filename}_{subset}.json"
            path_to_subset = Path(self.converter._output_dir, output_name)

            with path_to_subset.open("r") as f:
                subset_data = json.load(f)
            subset_count += len(subset_data["images"])

        self.assertEqual(subset_count, len(self.converter._input["images"]))

        print("Deleting dummy jsons if the test passes.")
        shutil.rmtree(self.converter._output_dir)

    def test__convert_data(self):

        self.converter._dimensions_to_int = Mock()
        self.converter._add_key = Mock()
        self.converter._to_absolute = Mock()
        self.converter._calculate_area = Mock()
        self.converter._split = Mock()

        m = Mock()
        m.configure_mock(one=self.converter._dimensions_to_int,
                         two=self.converter._add_key,
                         three=self.converter._to_absolute,
                         four=self.converter._calculate_area,
                         five=self.converter._split)

        self.converter.convert_data()
        m.assert_has_calls([call.one(key='iscrowd', value=0), call.two(), call.three(), call.four(), call.five()])


    #  OBSOLETE
    def test_hyperparameter_search(self):
        """
        params_info = {
            "x": [ 1, 2, 3],
            "y": [ "a", "b", "c"],
            "z": [ "A", "B", "C"],
            }
    
        for param_vals in itertools.product(*params_info.values()):
            params = dict(zip(params_info.keys(), param_vals))
            data = {
              "A": params["x"],
              "B": "Green",
              "C": {
                "c_a": "O2",
                "c_b": params["y"],
                "c_c": ["D", "E", "F", params["z"]]
              }
            }
            jsonstr = json.dumps(data) # use json.dump if you want to dump to a file
            print(jsonstr)
            
        """

        assert False
