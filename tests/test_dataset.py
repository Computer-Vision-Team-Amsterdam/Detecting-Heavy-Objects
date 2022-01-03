from unittest import TestCase

from detectron2.data import MetadataCatalog

from dataset import DATASET_NAME, register_dataset
from inference import plot_instance_segm
from utils import get_container_dicts


class Test(TestCase):
    def test_register_dataset(self) -> None:
        """
        This test checks whether a new dataset is succesfully registered in the dataset catalog from detectron2
        We plot some instances from the newly registered dataset to check if everything works as expected.
        """
        initial_catalog = MetadataCatalog.list()
        register_dataset(name=DATASET_NAME)
        extended_catalog = MetadataCatalog.list()
        actual_diff = list(set(extended_catalog) - set(initial_catalog))
        expected_diff = [f"{DATASET_NAME}_train", f"{DATASET_NAME}_val"]
        self.assertCountEqual(expected_diff, actual_diff)

        metadata = MetadataCatalog.get(f"{DATASET_NAME}_train")
        dataset_dicts = get_container_dicts("../data/train")
        plot_instance_segm(dataset_dicts, metadata, mode="ann", n_sample=3)
