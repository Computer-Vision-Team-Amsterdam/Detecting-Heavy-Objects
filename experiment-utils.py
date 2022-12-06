import json
from typing import NamedTuple

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import register_coco_instances

from utils_coco import add_images_to_coco, load_via_json


class ExperimentConfig(NamedTuple):
    """
    :param dataset_name: name of the dataset
    :param subset: what subset of data we visualize: train, val or test
    :param data_format: coco or via
    :param data_folder: name of the root folder where data is located
    """

    dataset_name: str
    subset: str
    data_format: str
    data_folder: str


def register_dataset(expCfg: ExperimentConfig) -> None:
    """
    Register dataset.
    """

    if expCfg.data_format == "coco":
        ann_path = f"{expCfg.data_folder}/{expCfg.subset}/containers-annotated-COCO-{expCfg.subset}.json"
        try:
            with open(ann_path) as f:
                _ = json.load(f)
                register_coco_instances(
                    f"{expCfg.dataset_name}_{expCfg.subset}",
                    {},
                    ann_path,
                    image_root=f"{expCfg.data_folder}",
                )
        except FileNotFoundError:
            if expCfg.subset == "test":
                add_images_to_coco(
                    image_dir=f"{expCfg.data_folder}/{expCfg.subset}",
                    coco_filename=f"{expCfg.data_folder}/{expCfg.subset}/containers-annotated-COCO-{expCfg.subset}.json",
                )
                ann_path = f"{expCfg.data_folder}/{expCfg.subset}/containers-annotated-COCO-{expCfg.subset}.json"
                register_coco_instances(
                    f"{expCfg.dataset_name}_{expCfg.subset}",
                    {},
                    ann_path,
                    image_root=f"{expCfg.data_folder}",
                )
            else:
                raise FileNotFoundError("No annotation file found")

        print(f"INFO:::{expCfg.dataset_name}_{expCfg.subset} has been registered!")
    if expCfg.data_format == "via":
        DatasetCatalog.register(
            f"{expCfg.dataset_name}_{expCfg.subset}",
            lambda d=expCfg.subset: load_via_json(f"{expCfg.data_folder}/" + d),
        )
        MetadataCatalog.get(f"{expCfg.dataset_name}_{expCfg.subset}").set(
            thing_classes=[f"{expCfg.dataset_name}"]
        )
