"""
This module contains functionality to register a new dataset
in Detectron2 dataset catalog
"""
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.utils.logger import setup_logger

from utils import get_container_dicts

setup_logger()

DATASET_NAME = "container"


def register_dataset(name: str) -> None:
    """
    Update detectron2 dataset catalog with our custom dataset.
    """

    for data_set in ["train", "val"]:
        DatasetCatalog.register(
            f"{name}_" + data_set, lambda d=data_set: get_container_dicts("data/" + d)
        )
        MetadataCatalog.get(f"{name}_" + data_set).set(thing_classes=[f"{name}"])
