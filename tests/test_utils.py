import json
import shutil
from pathlib import Path
from typing import Any, List
from unittest import TestCase
from unittest.mock import Mock, call

import pytest

pytest.importorskip("osr")
from utils_coco import DataFormatConverter  # pragma: no cover


class Test(TestCase):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super(Test, self).__init__(*args, **kwargs)

        input = Path("containers-annotated-1151-4000x2000.json")
        output_dir = "converter_output"

        script_location = Path(__file__).absolute().parent
        input = script_location / input
        self.converter = DataFormatConverter(input, output_dir)

    def test__add_key(self) -> None:

        # firstly, discard all images with bad resolution
        # make sure we resize the 2000x1000 images to 4000x2000, then re-create the input file for this test.
        self.converter._add_key("iscrowd", 0)
        for ann in self.converter._input["annotations"]:
            self.assertTrue(("iscrowd", 0) in ann.items())

    def test__calculate_area(self) -> None:
        self.converter._to_absolute()
        self.converter._calculate_area()
        for ann in self.converter._input["annotations"]:
            self.assertTrue(("area", 0) not in ann.items())

        """
        get_obj_size.plot_avg_bbox(data=self.converter._input,
                                   output_dir="/Users/dianaepureanu/Documents/Projects/Detecting-Heavy-objects/tests/")
        """

    def test__to_absolute(self) -> None:

        relative_segm: List[List[float]] = [
            ann["segmentation"][0] for ann in self.converter._input["annotations"]
        ]
        relative_bbox: List[List[float]] = [
            ann["bbox"] for ann in self.converter._input["annotations"]
        ]

        self.converter._to_absolute()

        absolute_segm: List[List[float]] = [
            ann["segmentation"][0] for ann in self.converter._input["annotations"]
        ]
        absolute_bbox: List[List[float]] = [
            ann["bbox"] for ann in self.converter._input["annotations"]
        ]

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

    def test__split(self) -> None:

        self.converter._split()

        subset_count = 0
        for subset in ["train", "val", "test"]:
            output_name = f"{self.converter._filename}_{subset}.json"
            path_to_subset = Path(self.converter._output_dir, output_name)

            with path_to_subset.open("r") as f:
                subset_data = json.load(f)
            f.close()
            subset_count += len(subset_data["images"])

        self.assertEqual(subset_count, len(self.converter._input["images"]))

        print("Deleting dummy jsons if the test passes.")
        shutil.rmtree(self.converter._output_dir)

    def test__convert_data(self) -> None:

        self.converter._dimensions_to_int = Mock()  # type: ignore
        self.converter._add_key = Mock()  # type: ignore
        self.converter._to_absolute = Mock()  # type: ignore
        self.converter._calculate_area = Mock()  # type: ignore
        self.converter._split = Mock()  # type: ignore

        m = Mock()
        m.configure_mock(
            one=self.converter._dimensions_to_int,
            two=self.converter._add_key,
            three=self.converter._to_absolute,
            four=self.converter._calculate_area,
            five=self.converter._split,
        )

        self.converter.convert_data()
        m.assert_has_calls(
            [
                call.one(),
                call.two(key="iscrowd", value=0),
                call.three(),
                call.four(),
                call.five(),
            ]
        )
