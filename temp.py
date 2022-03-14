import json
from tqdm import tqdm
import itertools

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
    # add code here to do something with json
"""


def size_to_int(input_path):
    with open(input_path) as json_file:
        data = json.load(json_file)

    for idx, image in tqdm(enumerate(data["images"]),
                           total=len(data["images"])):
        height = int(image["height"])
        width = int(image["width"])
        data["images"][idx]["height"] = height
        data["images"][idx]["width"] = width

    # save updated json
    with open('/Users/dianaepureanu/Documents/Projects/containers-annotated-COCO-integer-size.json', 'w') as f:
        json.dump(data, f)


def yolo_to_coco(input_path):
    """
    Converts Azure ML data labelling COCO format from percentages to absolute values.
    """
    with open(input_path) as json_file:
        data = json.load(json_file)

    for idx, image in tqdm(enumerate(data["images"]),
                           total=len(data["images"]),
                           desc="Convert percentages to COCO format"):
        height = image["height"]
        width = image["width"]
        image_id = image["id"]

        for ann_idx, annotation in enumerate(data["annotations"]):
            # use the corresponding width and height
            if annotation["image_id"] == image_id:
                segmentation = annotation["segmentation"][0]

                coco_segmentation = []
                for x, y in zip(segmentation[::2], segmentation[1::2]):
                    x_coco = round(x * width)
                    y_coco = round(y * height)
                    coco_segmentation.append(x_coco)
                    coco_segmentation.append(y_coco)

                # update input file
                data["annotations"][ann_idx]["segmentation"][0] = coco_segmentation

                bbox = annotation["bbox"]

                coco_bbox = []
                for x, y in zip(bbox[::2], bbox[1::2]):
                    x_coco = round(x * width)
                    y_coco = round(y * height)
                    coco_bbox.append(x_coco)
                    coco_bbox.append(y_coco)

                # update input file
                data["annotations"][ann_idx]["bbox"] = coco_bbox

    # save updated json
    with open('/Users/dianaepureanu/Documents/Projects/containers-annotated-COCO.json', 'w') as f:
        json.dump(data, f)


def split(input_path):
    """
    Split COCO json file in train val and test files.
    format:
    data, dict 3: images, annotations, categories
    images: list of dicts
    annotations: list of dicts
    categories: list of dicts

    """

    with open(input_path) as json_file:
        data = json.load(json_file)

    train_json = {"images": [], "annotations": [], "categories": data["categories"]}
    val_json = {"images": [], "annotations": [], "categories": data["categories"]}
    test_json = {"images": [], "annotations": [], "categories": data["categories"]}

    train_imgs_ids = []
    val_imgs_ids = []
    test_imgs_ids = []

    for image in data["images"]:
        filename = image["file_name"]
        id = image["id"]
        subset = filename.split("/")[0]

        if subset == "train":
            train_json["images"].append(image)
            train_imgs_ids.append(id)
        elif subset == "val":
            val_json["images"].append(image)
            val_imgs_ids.append(id)
        elif subset == "test":
            test_json["images"].append(image)
            test_imgs_ids.append(id)
        else:
            raise Exception(f"Unassigned image. Filename is {filename}.")

    for annotation in data["annotations"]:
        image_id = annotation["image_id"]

        if image_id in train_imgs_ids:
            train_json["annotations"].append(annotation)
        elif image_id in val_imgs_ids:
            val_json["annotations"].append(annotation)
        elif image_id in test_imgs_ids:
            test_json["annotations"].append(annotation)
        else:
            raise Exception("Unassigned annotation.")

    assert len(train_json["images"]) + len(val_json["images"]) + len(test_json["images"]) == len(data["images"])
    assert set(train_imgs_ids) ^ set(val_imgs_ids)
    assert set(val_imgs_ids) ^ set(test_imgs_ids)

    with open('/Users/dianaepureanu/Documents/Projects/train-containers-annotated-COCO.json', 'w') as f:
        json.dump(train_json, f)
    with open('/Users/dianaepureanu/Documents/Projects/val-containers-annotated-COCO.json', 'w') as f:
        json.dump(val_json, f)
    with open('/Users/dianaepureanu/Documents/Projects/test-containers-annotated-COCO.json', 'w') as f:
        json.dump(test_json, f)
