"""
This module contains general functionality to handle the annotated data
"""
import copy
import itertools
import json
import logging
import os
import shutil
from pathlib import Path
from typing import Any, Dict, List, NamedTuple, Tuple, Union

import cv2
import numpy as np
import pycocotools.mask as mask
import yaml
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import load_coco_json, register_coco_instances
from detectron2.structures import BoxMode
from PIL import Image


class DataFormatConverter:
    """
    Converts AzureML COCO format to Detectron2 COCO format.
    We encounter the following mismatches:

    1. Single vs separate files.

    AzureML generate one json file with annotations. images[i].file_name is named "${SUBSET}/${image_name}.jpg",
    where ${SUBSET} is {train, val, test}. This happens because of how we store the data in Azure Dataset, i.e. put them
    in 3 separate folders with named train, val and test. For Detectron2 training, we need 3 separate json files.
    We can split the single json file by searching all images from a given subset and then collecting the
    corresponding annotations. Images from the separate json files should also be renamed
    from "{$SUBSET}/image_name.jpg" to "image_name.jpg". We generate 3 separate json files.
    --------------------------------------------------
    2. Normalized coordinates vs absolute coordinates.

    In object detection projects, the exported "bbox": [x,y,width,height]" values in AzureML COCO file are normalized.
    They are scaled to 1. Example : a bounding box at (10, 10) location, with 30 pixels width , 60 pixels height,
    in a 640x480 pixel image will be annotated as (0.015625. 0.02083, 0.046875, 0.125).
    Since the coordintes are normalized, it will show as '0.0' as "width" and "height" for all images.
    The same thing happens to the segmentation values. We convert and update these values.
    --------------------------------------------------
    3. Miscalculation of area

    AzureML COCO file contains area of 0 for all images. Instead, it should contain the area of the segmentation mask,
    i.e. the area described by the instance segmentation polygon. We compute and update the area.
    --------------------------------------------------
    4. Missing keys

    AzureML COCO file does not contain the "iscrowd" key in the annotations, which causes errors at evaluation time.
    Since we are not doing a crowd counting task, the "iscrowd" key will always be 0. We add it to the annotation
    file.
    --------------------------------------------------
    [5. Image dimensions are stored as strings instead of ints] - if we start from
    containers-annotated-1151.json, which is already partially processed,
    this step is not necessary since it was already done.
    If we download another json file from AzureML, it will be the case that the dimensions are strings.
    """

    def __init__(self, azureml_file: Path, output_dir: str):
        """
        Args:
            azureml_file: json file that was generated by the Data labelling tool in AzureML.

                We assume the data in Azure is stored in train, val and test folders and images have
                as filename the following: "${SUBSET}/${image_name}.jpg", where ${SUBSET} is {train, val, test}

            output_dir: an output directory to dump the converted json files.


        """
        self._logger = logging.getLogger(__name__)
        self._filename = Path(azureml_file).stem  # extract the filename given a path
        self._output_dir = output_dir

        with open(azureml_file) as f:
            self._input = json.load(f)
        f.close()

        """
        Calculation of polygon area is affected by image resolution. When we convert
        polygon points to RLE object, we use height = 2000 and width = 4000 
        """

        for image in self._input["images"]:
            """
            assert isinstance(image["height"], float) and isinstance(image["width"], float), \
                f'Image dimensions must be integers,' \
                f'found {type(image["height"])} x {type(image["width"])} instead!'
            """
            assert image["height"] == 2000 and image["width"] == 4000, (
                f"Image resolution should be 2000x4000,"
                f'found resolution {image["height"]} x {image["width"]} instead!'
            )

    def _dimensions_to_int(self) -> None:
        """
        containers-annotated-1151.json has floats, but original AzureML output has strings;
        I added this method to make sure the dimensions are in the right type, but if
        they are floats, we do not need to call this function.
        """
        for i, _ in enumerate(self._input["images"]):
            self._input["images"][i]["width"] = int(self._input["images"][i]["width"])
            self._input["images"][i]["height"] = int(self._input["images"][i]["height"])

    def _add_key(self, key: str, value: int) -> None:
        """
        Adds key to the input file
        key: name of key to be added in _input[annotations]
        value: corresponding value for the key
        """
        for i, _ in enumerate(self._input["annotations"]):
            self._input["annotations"][i][key] = value

    def _calculate_area(self) -> None:
        """
        Updates area based on polygon coordinate
        """

        def PolyArea(x: List[float], y: List[float]) -> np.float64:
            return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))  # type: ignore

        """
        The deep copy creates independent copy of original object and all its nested objects.
        A shallow copy doesn't create a copy of nested objects, instead it just copies 
        the reference of nested objects. This means, a copy process does not recurse or 
        create copies of nested objects itself. If you change a nested element in a shallow
        copy, it will also get changed by reference in the original object.
        
        We want the polygon segmentation in the output files, so we use a copy for the RLE format.
        """
        anns_copy = copy.deepcopy(self._input["annotations"])

        for i, ann in enumerate(anns_copy):
            xs = [value for i, value in enumerate(ann["segmentation"][0]) if i % 2 == 0]
            ys = [value for i, value in enumerate(ann["segmentation"][0]) if i % 2 == 1]

            area = PolyArea(xs, ys)
            self._input["annotations"][i]["area"] = area

    def _to_absolute(self) -> None:
        """
        Converts normalized bbox and segmentation values to absolute values.
        All images must have the same width and height.
        """
        width = self._input["images"][0]["width"]
        height = self._input["images"][0]["height"]

        for i, ann in enumerate(self._input["annotations"]):
            absolute_values = []
            # iterate through every pair of 2 elements
            for x, y in zip(ann["segmentation"][0][::2], ann["segmentation"][0][1::2]):
                absolute_values.append(x * width)
                absolute_values.append(y * height)
            self._input["annotations"][i]["segmentation"][0] = absolute_values

            bbox_absolute_values = []
            for x, y in zip(ann["bbox"][::2], ann["bbox"][1::2]):
                bbox_absolute_values.append(x * width)
                bbox_absolute_values.append(y * height)
            self._input["annotations"][i]["bbox"] = bbox_absolute_values

    def _split(self) -> None:
        """
        Splits input file in train, val and test json files.
        Stores the 3 files in the output_dir.
        """

        def _collect_images(subset: str) -> Tuple[List[Dict[str, Any]], List[str]]:
            """
            Collect all images that belong to a subset
            We need their ids as well since we collect annotations based on the ids.
            """
            ids: List[str] = []
            images: List[Dict[str, Any]] = []
            for image in self._input["images"]:
                image_subset, image_name = image["file_name"].split("/")
                if image_subset == subset:
                    images.append(image)
                    ids.append(image["id"])
            return images, ids

        def _collect_annotations(ids: List[str]) -> List[Any]:
            """
            Collect all annotations that belong to a subset
            """
            annotations = []
            for ann in self._input["annotations"]:
                if ann["image_id"] in ids:
                    annotations.append(ann)
            return annotations

        subsets = ["train", "val", "test"]
        for subset in subsets:
            subset_images, subset_ids = _collect_images(subset)
            subset_annotations = _collect_annotations(subset_ids)
            output = {
                "images": subset_images,
                "annotations": subset_annotations,
                "categories": self._input["categories"],
            }

            # create output directory structure if it does not exist already
            Path(self._output_dir).mkdir(parents=True, exist_ok=True)
            output_name = f"{self._filename}_{subset}.json"

            self._logger.info(
                f"Storing {subset} json at {Path(self._output_dir, output_name)}."
            )
            with open(Path(self._output_dir, output_name), "w") as f:
                json.dump(output, f)
            f.close()

    def convert_data(self) -> None:
        self._dimensions_to_int()
        self._add_key(key="iscrowd", value=0)
        self._to_absolute()  # we must calculate area based on absolute values
        self._calculate_area()
        self._split()


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


def get_container_dicts(
    expCfg: ExperimentConfig,
) -> List[Dict[str, Any]]:
    """
    Return annotations in json format

    """

    if expCfg.data_format == "coco":
        container_dicts = load_coco_json(
            f"{expCfg.data_folder}/{expCfg.subset}/containers-annotated-COCO-{expCfg.subset}.json",
            image_root=f"{expCfg.data_folder}",
        )
    elif expCfg.data_format == "via":
        container_dicts = load_via_json(f"{expCfg.data_folder}/{expCfg.subset}")
    else:
        raise Exception("Wrong data format")

    return container_dicts  # type: ignore


def load_via_json(img_dir: Union[Path, str]) -> List[Dict[str, Any]]:
    """
    Parse annotations json

    :param img_dir: path to directory with images and annotations
    :return: images metadata
    """

    json_file = os.path.join(img_dir, "containers-annotated.json")
    with open(json_file) as file:
        imgs_anns = json.load(file)

        try:
            imgs_anns = imgs_anns["_via_img_metadata"]
        except KeyError:
            print("Annotation file has no metadata.")
    file.close()
    dataset_dicts = []
    for idx, value in enumerate(imgs_anns.values()):
        record = {}  # type: Dict[str, Any]

        filename = os.path.join(img_dir, value["filename"])
        height, width = cv2.imread(filename).shape[:2]

        record["file_name"] = filename
        record["image_id"] = idx
        record["height"] = height
        record["width"] = width

        annos = value["regions"]
        objs = []
        for anno in annos:
            # assert not anno["region_attributes"]
            anno = anno["shape_attributes"]
            p_x = anno["all_points_x"]
            p_y = anno["all_points_y"]
            poly = [(x + 0.5, y + 0.5) for x, y in zip(p_x, p_y)]
            poly = [p for x in poly for p in x]

            obj = {
                "bbox": [np.amin(p_x), np.amin(p_y), np.amax(p_x), np.amax(p_y)],
                "bbox_mode": BoxMode.XYXY_ABS,
                "segmentation": [poly],
                "category_id": 0,
            }
            objs.append(obj)
        record["annotations"] = objs
        dataset_dicts.append(record)
    return dataset_dicts


def collect_nested_lists(
    dictionary: Dict[str, Any],
    composed_key: str,
    nested_lists: Dict[str, List[str]],
) -> Dict[str, List[str]]:
    """
    This method parses a nested dictionary recursively and collects the (composed) keys where the value is a list.
    :param dictionary:
    :param composed_key:
    :param nested_lists:

    :return: keys and values of the @dictionary such that values are of type list.
    """

    for k, v in dictionary.items():
        if isinstance(v, dict):
            if composed_key == "":
                collect_nested_lists(v, k, nested_lists)
            else:
                collect_nested_lists(v, composed_key + "." + k, nested_lists)
        if isinstance(v, list):
            if composed_key == "":
                nested_lists[k] = v
            else:
                nested_lists[composed_key + "." + k] = v

    return nested_lists


def generate_config_file(file: Any, configuration: Dict[Any, Any], name: int) -> None:
    for composed_name, value in configuration.items():
        names = composed_name.split(".")
        if len(names) == 1:
            file[names[0]] = value
        if len(names) == 2:
            file[names[0]][names[1]] = value
        if len(names) == 3:
            file[names[0]][names[1]][names[2]] = value

    with open(f"configs/temp_{name}.yaml", "w") as outfile:
        yaml.dump(file, outfile, sort_keys=False)
    outfile.close()


def handle_hyperparameters(config: Union[str, Path]) -> int:
    # open yaml file as dict
    with open(config) as f:
        file = yaml.safe_load(f)
    f.close()
    # get rows for which we do hyperparameter search
    grid_space = collect_nested_lists(file, "", {})

    count = 0
    for combination in itertools.product(*grid_space.values()):
        configuration = dict(zip(grid_space.keys(), combination))
        generate_config_file(file, configuration, count)
        count = count + 1

    return count


def register_dataset(expCfg: ExperimentConfig) -> None:
    """
    Register dataset.
    """
    if expCfg.data_format == "coco":
        ann_path = f"{expCfg.data_folder}/{expCfg.subset}/containers-annotated-COCO-{expCfg.subset}.json"
        register_coco_instances(
            f"{expCfg.dataset_name}_{expCfg.subset}",
            {},
            ann_path,
            image_root=f"{expCfg.data_folder}",
        )
    if expCfg.data_format == "via":
        DatasetCatalog.register(
            f"{expCfg.dataset_name}_{expCfg.subset}",
            lambda d=expCfg.subset: load_via_json(f"{expCfg.data_folder}/" + d),
        )
        MetadataCatalog.get(f"{expCfg.dataset_name}_{expCfg.subset}").set(
            thing_classes=[f"{expCfg.dataset_name}"]
        )


def correct_faulty_panoramas() -> None:
    """
    When creating the initial train, val and test sets, we had images of 2 different
    resolutions, 2000x1000 and 4000x2000. This caused problems since the annotations
    were different and we could not easily define what is Small, Medium or Large container.
    The images in 2000x1000 resolution were downloaded from the panorama API using an older version,
    so they have filename as their id instead of panorama_id.

    We need to:
     - look at the 2000x1000 panoramas, try to query for them on the API (which is not possible
       based on filename, so we will look through images until we find a match),
     - download the images with panorama_ids as their id and in a 4000x2000 resolution,
     - replace them in the train, val and test sets
     - update images height and width in the annotation file

     (- recalculate the segmentation and bbox. This is simple since the file from AzureML
     already has normalized values, so we will just multiply by H=2000 and W=4000.)
     This last step is not necessary since we can apply denormalization to all images when we do the conversion
     to proper COCO format now that we know all images have the same resolution.

    """

    data_root = "/Users/dianaepureanu/Documents/Projects/versions_of_data/data_extended"
    input = "/Users/dianaepureanu/Documents/Projects/versions_of_data/data_extended/annotations-renamed-filenames.json"
    resized_input = "/Users/dianaepureanu/Documents/Projects/versions_of_data/data_extended/annotations-renamed-filenames-4000x2000.json"
    output_dir = "/Users/dianaepureanu/Documents/Projects/versions_of_data/data_extended/4000x2000"
    subsets = ["train", "val", "test"]

    with open(input) as f:
        _input = json.load(f)
    f.close()

    def _create_dirs() -> None:
        """
        Create folder structures where we put original and upscaled images
        """
        for subset in subsets:
            Path(output_dir + f"/{subset}_2000x1000").mkdir(parents=True, exist_ok=True)
            Path(output_dir + f"/{subset}_4000x2000").mkdir(parents=True, exist_ok=True)

    def _copy(_input: Dict[str, Any]) -> None:
        """
        Find images of resolution 2000x1000 and copy them to a different directory
        """
        for image in _input["images"]:
            if int(image["width"]) == 2000:
                subset, filename = image["file_name"].split("/")

                # get image in subset
                if subset in ["train", "val", "test"]:
                    panoramas = Path(data_root, subset).glob("*.jpg")
                    for pano in panoramas:
                        pano_filename = pano.parts[-1]
                        if pano_filename == filename:
                            shutil.copy(pano, output_dir + f"/{subset}_2000x1000")
                else:
                    raise Exception("Subset must be either train, val or test!")

    def _is_copied() -> bool:
        """
        Check whether 2000x1000 files are already copied in separate folders
        """
        copied = True
        for subset in subsets:
            low_res_images = Path(output_dir + f"/{subset}_2000x1000").glob("*.jpg")
            elements_count = sum(1 for _ in low_res_images)
            if elements_count == 0:
                print(f"Copying images of dimension 2000x1000 to {subset}")
                copied = False
        return copied

    def _upscale(image_path: str) -> Any:
        """
        Upscale one image
        """

        img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

        # print('Original Dimensions : ', img.shape)

        scale_percent = 200  # percent of original size
        width = int(img.shape[1] * scale_percent / 100)
        height = int(img.shape[0] * scale_percent / 100)
        dim = (width, height)

        # resize image
        resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

        # print('Resized Dimensions : ', resized.shape)

        return resized

    def _upscale_all() -> None:
        """
        Upscale all images and store them in the 4000x2000 folders
        """
        for subset in subsets:
            images_to_upscale = Path(output_dir + f"/{subset}_2000x1000").glob("*.jpg")

            for image in images_to_upscale:
                upscaled = _upscale(str(image))

                path = Path(output_dir, f"{subset}_4000x2000/" + image.parts[-1])

                cv2.imwrite(str(path), upscaled)
                # upscaled.save(path)

    def _is_upscaled() -> bool:
        """
        Check whether 2000x1000 files are already upscaled
        """
        upscaled = True
        for subset in subsets:
            low_res_images = Path(output_dir + f"/{subset}_4000x2000").glob("*.jpg")
            elements_count = sum(1 for _ in low_res_images)
            if elements_count == 0:
                upscaled = False
        return upscaled

    def _update_dims() -> None:
        """
        Update dimensions in the original json file from 2000x1000 to 4000x2000
        """

        for i, image in enumerate(_input["images"]):
            if image["width"] == 2000:
                _input["images"][i]["width"] = 4000
                _input["images"][i]["height"] = 2000

        with open(resized_input, "w") as f:
            json.dump(_input, f)
        f.close()

    _create_dirs()
    if _is_copied() is False:
        print(f"Copying images to {output_dir}/$subset_2000x1000...")
        _copy(_input)
    else:
        print(f"Images were already copied to {output_dir}/$subset_2000x1000...")

    if _is_upscaled() is False:
        print(f"Upscaling images and storing them at {output_dir}/$subset_4000x2000...")
        _upscale_all()
    else:
        print(
            f"Images were already upscaled and stored at {output_dir}/$subset_4000x2000..."
        )

    _update_dims()


def collect_dimensions(data: Any) -> Tuple[List[int], List[int]]:
    """
    Collects widths and heights from json file
    """
    # assert if we have a list/results json or a dict/annotations json
    if isinstance(data, list):  # this is a results json file
        pass
    if isinstance(data, dict):  # this is an annotation json file
        data = data["annotations"]

    widths = []
    heights = []
    for ann in data:
        width = ann["bbox"][2]
        height = ann["bbox"][3]

        widths.append(int(width))
        heights.append(int(height))

    return widths, heights


# correct_faulty_panoramas()

"""
input = "/Users/dianaepureanu/Documents/Projects/versions_of_data/data_extended/annotations-renamed-filenames-4000x2000.json"
output_dir = "/Users/dianaepureanu/Documents/Projects/Detecting-Heavy-objects/tests/converter_output_data_extended"
converter = DataFormatConverter(input, output_dir)
converter.convert_data()
"""