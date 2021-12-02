from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.utils.logger import setup_logger

from inference import visualize_images
from utils import get_container_dicts

setup_logger()


def register_dataset(name: str) -> None:
    """
    Update detectron2 dataset catalog with our custom dataset.
    """

    for d in ["train", "val"]:
        DatasetCatalog.register(
            f"{name}_" + d, lambda d=d: get_container_dicts("data/" + d)
        )
        MetadataCatalog.get(f"{name}_" + d).set(thing_classes=[f"{name}"])


if __name__ == "__main__":

    dataset_name = "container"
    register_dataset(name=dataset_name)
    metadata = MetadataCatalog.get(f"{dataset_name}_train")
    dataset_dicts = get_container_dicts("data/train")
    visualize_images(dataset_dicts, metadata, mode="ann", n_sample=3)
