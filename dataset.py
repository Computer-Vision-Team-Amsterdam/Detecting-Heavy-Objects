"""
This module contains functionality to register a new dataset
in Detectron2 dataset catalog
"""
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.utils.logger import setup_logger

from utils import load_via_json
from detectron2.data.datasets import register_coco_instances
setup_logger()


def register_dataset(name: str, data_format: str) -> None:
    """
    @param name: name of the dataset
    @param data_format: format of the dataset. Choices: coco, via
    """
    for subset in ["train", "val"]:
        if data_format == "coco":
            ann_path = f"data/{subset}/containers-annotated-COCO-{subset}.json"
            register_coco_instances(f"{name}_{subset}", {}, ann_path, image_root="data")
        if data_format == "via":
            DatasetCatalog.register(f"{name}_{subset}", lambda d=subset: load_via_json("data/" + d))
            MetadataCatalog.get(f"{name}_{subset}").set(thing_classes=[f"{name}"])

