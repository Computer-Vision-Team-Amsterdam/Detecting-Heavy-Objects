from pathlib import Path
from unittest import TestCase

import pytest

pytest.importorskip("osr")

from utils import collect_nested_lists

ROOT = Path(__file__).parent.parent


class Test(TestCase):
    def test_collect_nested_lists(self) -> None:
        data = {
            "A": "Blue",
            "B": ["Green", "Red"],
            "C": {"c_a": "O2", "c_b": "05", "c_c": ["D", "E", "F", "f"]},
        }

        expected_result = {"B": ["Green", "Red"], "C.c_c": ["D", "E", "F", "f"]}

        actual_result = collect_nested_lists(data, "", {})
        print(actual_result)
        self.assertDictEqual(expected_result, actual_result)
