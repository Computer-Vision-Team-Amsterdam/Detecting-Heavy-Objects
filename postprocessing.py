"""
This model applies different post processing steps on the results file of the detectron2 model
"""
import json
from pathlib import Path
from typing import List

from visualizations.stats import DataStatistics


class PostProcessing:
    """
    Post processing operations on the output of the predictor
    """

    def __init__(
            self,
            json_predictions: Path,
            threshold: float = 20000,
            output_path: Path = Path.cwd() / "postprocessed.json",
    ) -> None:
        """
        Args:
            :param json_predictions: path to ouput file with predictions from detectron2
            :param threshold: objects with a bbox smaller and equal to this arg are discarded. value is in pixels
            :param output_path: where the filtered json with predictions is stored
        """
        self.stats = DataStatistics(json_file=json_predictions)
        self.threshold = threshold
        self.output_path = output_path
        self.predictions_to_keep: List[str] = []

    def filter_by_size(self) -> None:
        """
        Removes predictions of small objects and writes results to json file
        """

        def remove_predictions() -> None:
            """
            Filter out all predictions where area of the bbox of object is smaller and equal to @param threshold pixels
            """
            indices_to_keep = [
                idx
                for idx, area in enumerate(self.stats.areas)
                if area > self.threshold
            ]
            self.predictions_to_keep = [self.stats.data[idx] for idx in indices_to_keep]

        def write_json() -> None:
            """
            Write filtered list of predictions to another json file
            """
            with open(self.output_path, "w") as f:
                json.dump(self.predictions_to_keep, f)
            f.close()

        remove_predictions()
        write_json()
        print(self.output_path)


def discard_annotations():
    train_json = "/Users/dianaepureanu/Documents/Projects/versions_of_annotations/containers-annotated-COCO-train.json"
    val_json = "/Users/dianaepureanu/Documents/Projects/versions_of_annotations/containers-annotated-COCO-val.json"
    test_json = "/Users/dianaepureanu/Documents/Projects/versions_of_annotations/containers-annotated-COCO-test.json"

    threshold = 8000

    with open(train_json) as f:
        train = json.load(f)
    f.close()
    with open(val_json) as f:
        val = json.load(f)
    f.close()
    with open(test_json) as f:
        test = json.load(f)
    f.close()

    # create new annotation files
    new_train_anns = {"images": [], "annotations": [], "categories": [{"id": 1, "name": "container"}]}
    new_val_anns = {"images": [], "annotations": [], "categories": [{"id": 1, "name": "container"}]}
    new_test_anns = {"images": [], "annotations": [], "categories": [{"id": 1, "name": "container"}]}

    # append to new annotation files
    # TRAIN
    for image in train["images"]:
        id = image["id"]
        found = False
        for ann in train["annotations"]:
            image_id = ann["image_id"]
            if image_id == id:
                # check annotation area
                if ann["area"] >= threshold:
                    found = True
                    # add annotation to list of annotations
                    new_train_anns["annotations"].append(ann)
            # this image has at least one annotation with a large enough area
            if found:
                # add image to the list of images
                new_train_anns["images"].append(image)

    # VAL
    for image in val["images"]:
        id = image["id"]
        found = False
        for ann in val["annotations"]:
            image_id = ann["image_id"]
            if image_id == id:
                # check annotation area
                if ann["area"] >= threshold:
                    found = True
                    # add annotation to list of annotations
                    new_val_anns["annotations"].append(ann)
            # this image has at least one annotation with a large enough area
            if found:
                # add image to the list of images
                new_val_anns["images"].append(image)

    # TEST
    for image in test["images"]:
        id = image["id"]
        found = False
        for ann in test["annotations"]:
            image_id = ann["image_id"]
            if image_id == id:
                # check annotation area
                if ann["area"] >= threshold:
                    found = True
                    # add annotation to list of annotations
                    new_test_anns["annotations"].append(ann)
            # this image has at least one annotation with a large enough area
            if found:
                # add image to the list of images
                new_test_anns["images"].append(image)

    # save new annotation files
    with open("threshold_8000/containers-annotated-COCO-train.json", "w") as f:
        json.dump(new_train_anns, f)
    with open("threshold_8000/containers-annotated-COCO-val.json", "w") as f:
        json.dump(new_val_anns, f)
    with open("threshold_8000/containers-annotated-COCO-test.json", "w") as f:
        json.dump(new_test_anns, f)
