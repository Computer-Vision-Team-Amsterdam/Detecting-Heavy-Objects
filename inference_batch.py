import argparse
import glob
import json
import os
from pathlib import Path
from typing import Any, List, Tuple

import detectron2.data.transforms as T
import numpy as np
import pycocotools.mask as mask_util
import torch
from cv2 import imread
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.modeling import build_model
from detectron2.structures import BoxMode
from psycopg2.extras import execute_values
from torch.utils.data import DataLoader, Dataset

import upload_to_postgres
from utils.azure_storage import StorageAzureClient
from utils.date import get_start_date


class ContainerDataset:
    def __init__(self, img_names: List[str], cfg: Any = None) -> None:
        self.img_names = img_names
        if len(cfg.DATASETS.TEST):
            self.metadata = MetadataCatalog.get(cfg.DATASETS.TEST[0])
        self.aug = T.ResizeShortestEdge(
            [cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST], cfg.INPUT.MAX_SIZE_TEST
        )  # TODO validate the input dims
        self.input_format = cfg.INPUT.FORMAT

    def __getitem__(self, index: int) -> Tuple[Any, str, Tuple[int]]:
        img = imread(self.img_names[index])
        if self.input_format == "RGB":
            # whether the model expects BGR inputs or RGB
            img = img[:, :, ::-1]
        image = self.aug.get_transform(img).apply_image(img)
        image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))

        img_name = self.img_names[index]

        return image, img_name, img.shape[:2]

    def __len__(self) -> int:
        return len(self.img_names)


def instances_to_coco_json(
    instances: Any, img_name: str
) -> Tuple[List[Any], List[Any]]:
    """
    Dump an "Instances" object to a COCO-format json that's used for evaluation.

    Args:
        instances (Instances):
        img_id (int): the image id

    Returns:
        list[dict]: list of json annotations in COCO format.
    """
    num_instance = len(instances)
    if num_instance == 0:
        return [], []

    boxes = instances.pred_boxes.tensor.numpy()
    boxes = BoxMode.convert(boxes, BoxMode.XYXY_ABS, BoxMode.XYWH_ABS)
    boxes = boxes.tolist()
    scores = instances.scores.tolist()
    classes = instances.pred_classes.tolist()

    has_mask = instances.has("pred_masks")
    if has_mask:
        # use RLE to encode the masks, because they are too large and takes memory
        # since this evaluator stores outputs of the entire dataset
        rles = [
            mask_util.encode(np.array(mask[:, :, None], order="F", dtype="uint8"))[0]
            for mask in instances.pred_masks
        ]
        for rle in rles:
            # "counts" is an array encoded by mask_util as a byte-stream. Python3's
            # json writer which always produces strings cannot serialize a bytestream
            # unless you decode it. Thankfully, utf-8 works out (which is also what
            # the pycocotools/_mask.pyx does).
            rle["counts"] = rle["counts"].decode("utf-8")

    has_keypoints = instances.has("pred_keypoints")
    if has_keypoints:
        keypoints = instances.pred_keypoints

    results_json = []
    results = []
    for k in range(num_instance):
        result = {
            "pano_id": img_name,
            "category_id": classes[k],
            "bbox": boxes[k],
            "score": scores[k],
        }
        if has_mask:
            result["segmentation"] = rles[k]
        else:
            # Don't append result dict when there is no segmentation mask
            print(f"Detectron2 had issues with instance segmentation task for image {img_name}.")
            continue
        if has_keypoints:
            # In COCO annotations,
            # keypoints coordinates are pixel indices.
            # However our predictions are floating point coordinates.
            # Therefore we subtract 0.5 to be consistent with the annotation format.
            # This is the inverse of data loading logic in `datasets/coco.py`.
            keypoints[k][:, :2] -= 0.5
            result["keypoints"] = keypoints[k].flatten().tolist()
        results_json.append(result)
        results.append([img_name, scores[k], boxes[k]])
    return results_json, results


def get_chunk_pano_ids():
    # Download txt file(s) with pano ids that we want to download from CloudVPS
    local_file = f"{opt.worker_id}.txt"
    saClient.download_blob(
        cname="retrieve-images-input",
        blob_name=f"{start_date_dag}/{local_file}",
        local_file_path=local_file,
    )
    with open(local_file, "r") as f:
        pano_ids = [line.rstrip("\n") for line in f]

    if len(pano_ids) < opt.num_workers:
        raise ValueError("Number of workers is larger than items to process. Aborting...")
    print(f"Printing first and last file names from the chunk: {pano_ids[0]} {pano_ids[-1]}")

    return pano_ids


def download_panos():
    # Get all file names of the panoramic images from the storage account
    blobs = saClient.list_container_content(
        cname="unblurred", blob_prefix=start_date_dag
    )

    # Validate if all blobs are available
    if len(set(pano_ids) - set(blobs)) != 0:
        raise ValueError("Not all panoramic images are available in the storage account! Aborting...")

    for blob in pano_ids:
        saClient.download_blob(
            cname="blurred",
            blob_name=f"{start_date_dag}/{blob}",
            local_file_path=f"{input_path}/{blob}",
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--date", type=str, help="Processing date in the format %Y-%m-%d %H:%M:%S.%f"
    )
    parser.add_argument("--device", type=str, help="Processing on CPU or GPU?")
    parser.add_argument("--weights", type=str, help="Trained weights filename")
    parser.add_argument("--worker-id", type=int, help="worker ID")
    parser.add_argument("--num-workers", type=int, help="number of workers")
    opt = parser.parse_args()

    results_file_name = f"coco_instances_results_{opt.worker_id}.json"

    start_date_dag, start_date_dag_ymd = get_start_date(opt.date)

    input_path = Path("images", start_date_dag_ymd)
    if not input_path.exists():
        input_path.mkdir(exist_ok=True, parents=True)

    # Download images from storage account
    saClient = StorageAzureClient(secret_key="data-storage-account-url")

    pano_ids = get_chunk_pano_ids()
    download_panos()

    cfg = get_cfg()
    cfg.merge_from_file("configs/container_detection.yaml")
    cfg.DATALOADER.NUM_WORKERS = 1
    cfg.MODEL.DEVICE = opt.device
    BATCH_SIZE = 1

    model = build_model(cfg)
    DetectionCheckpointer(model).load(opt.weights)
    model.train(False)

    image_names = [file_name for file_name in glob.glob(f"{input_path}/*.jpg")]
    dataset = ContainerDataset(img_names=image_names, cfg=cfg)

    data_loader = DataLoader(
        dataset=dataset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True
    )

    num = 0
    data_results_json = []
    data_results = []
    with torch.no_grad():
        for (imgs, img_names, shapes) in data_loader:
            inputs = []
            for i, img_tensor in enumerate(imgs):
                inputs.append(
                    {"image": img_tensor, "height": shapes[0][i], "width": shapes[1][i]}
                )

            all_outputs = model(inputs)

            for j, outputs in enumerate(all_outputs):
                if "instances" in outputs:
                    instances = outputs["instances"]
                    prediction_json, prediction = instances_to_coco_json(
                        instances, os.path.basename(img_names[j])
                    )
                if prediction_json and prediction:
                    data_results_json.append(prediction_json[0])
                    data_results.append(prediction[0])
                    num += 1

    print("Detect %d frames with objects in haul %s" % (num, input_path))

    if data_results_json:
        with open(results_file_name, "w") as f:
            json.dump(data_results_json, f)

        print("Upload detection file to Blob Storage...")
        saClient.upload_blob(
            cname="detections",
            blob_name=f"{start_date_dag}/{results_file_name}",
            local_file_path=results_file_name,
        )

    if data_results:
        with upload_to_postgres.connect() as (conn, cur):
            print("Inserting data into database...")
            table_name = "detections"

            # Get columns
            sql = f"SELECT * FROM {table_name} LIMIT 0"
            cur.execute(sql)
            table_columns = [desc[0] for desc in cur.description]
            table_columns.pop(0)  # Remove the id column

            # Inserting data into database
            query = f"INSERT INTO {table_name} ({','.join(table_columns)}) VALUES %s"
            execute_values(cur, query, data_results)
            conn.commit()
