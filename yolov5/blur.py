import argparse
import os
from datetime import datetime
from pathlib import Path

import torch
from models.experimental import attempt_load
from PIL import Image, ImageDraw, ImageFilter
from tqdm import tqdm

from utils.datasets import create_dataloader
from utils.general import (
    check_file,
    check_img_size,
    non_max_suppression,
    scale_coords,
    set_logging,
)
from utils.torch_utils import select_device

# TODO kan dit mooier?
import sys
import os
module_path = os.path.abspath(os.path.join("../utils"))
if module_path not in sys.path:
    sys.path.insert(0, module_path)
from utils.azure_storage import BaseAzureClient, StorageAzureClient
azClient = BaseAzureClient()


def blur_imagery(
    weights=None, batch_size=1, imgsz=640, conf_thres=0.001, iou_thres=0.6
):

    set_logging()

    if not opt.output_folder.exists():
        opt.output_folder.mkdir(exist_ok=True, parents=True)

    device = select_device(
        "cuda:0" if torch.cuda.is_available() else "cpu", batch_size=batch_size
    )
    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    imgsz = check_img_size(imgsz, s=model.stride.max())  # check img_size

    # Half
    half = device.type != "cpu"  # half precision only supported on CUDA
    if half:
        model.half()

    # Configure
    model.eval()

    print(f"reading from the following folder: {opt.folder}")
    print(f"contents of this folder: {os.listdir(Path(os.getcwd(), opt.folder))}")
    dataloader = create_dataloader(
        opt.folder, imgsz, batch_size, model.stride.max(), opt, pad=0.5, rect=True
    )[0]

    for batch_i, (img, paths, shapes) in enumerate(tqdm(dataloader)):
        img = img.to(device, non_blocking=True)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0

        with torch.no_grad():
            inf_out, train_out = model(img)  # inference and training outputs
            output = non_max_suppression(
                inf_out, conf_thres=conf_thres, iou_thres=iou_thres
            )

        # Statistics per image
        for si, pred in enumerate(output):
            # Predictions
            predn = pred.clone()
            scale_coords(
                img[si].shape[1:], predn[:, :4], shapes[si][0], shapes[si][1]
            )  # native-space pred
            image_path = Path(paths[si])
            img_file = Image.open(image_path)
            mask = Image.new("L", img_file.size, 0)
            draw = ImageDraw.Draw(mask)
            for *xyxy, conf, cls in predn.tolist():
                draw.rectangle(xyxy, fill=255)

            # Blur image
            blurred = img_file.filter(ImageFilter.GaussianBlur(52))
            # Paste blurred region and save result
            img_file.paste(blurred, mask=mask)
            img_file.save(opt.output_folder / image_path.name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="blur.py")
    parser.add_argument(
        "--weights",
        type=str,
        default=f"{os.getcwd()}/weights/best.pt",
        help="model.pt path(s)",
    )
    parser.add_argument(
        "--data", type=str, default=f"{os.getcwd()}/data/pano.yaml", help="*.data path"
    )
    parser.add_argument(
        "--folder",
        type=Path,
        default=f"{os.getcwd()}/unblurred",
        help="folder with images to blur",
    )
    parser.add_argument(
        "--output_folder",
        type=Path,
        default=f"{os.getcwd()}/blurred",
        help="Location where blurred images are stored",
    )
    parser.add_argument(
        "--batch-size", type=int, default=8, help="size of each image batch"
    )
    parser.add_argument(
        "--img-size", type=int, default=2048, help="inference size (pixels)"
    )
    parser.add_argument(
        "--conf-thres", type=float, default=0.25, help="object confidence threshold"
    )
    parser.add_argument(
        "--iou-thres", type=float, default=0.6, help="IOU threshold for NMS"
    )
    parser.add_argument("--date", type=str, help="date for images to be blurred")

    opt = parser.parse_args()
    opt.data = check_file(opt.data)  # check file
    print(opt)

    # Start date, string of form %Y-%m-%d %H:%M:%S.%f
    start_date = datetime.strptime(opt.date, "%Y-%m-%d %H:%M:%S.%f")
    my_format = "%Y-%m-%d_%H-%M-%S"  # Only use year month day format
    start_date_dag = start_date.strftime(my_format)

    # update input folder
    opt.folder = Path(opt.folder, start_date_dag)
    if not opt.folder.exists():
        opt.folder.mkdir(exist_ok=True, parents=True)

    # update output folder
    opt.output_folder = Path(opt.output_folder, start_date_dag)
    if not opt.output_folder.exists():
        opt.output_folder.mkdir(exist_ok=True, parents=True)

    # download images from storage account
    saClient = StorageAzureClient(secret_key="data-storage-account-url")
    blobs = saClient.list_container_content(
        cname="unblurred", blob_prefix=start_date_dag
    )
    for blob in blobs:
        blob = blob.split("/")[-1]  # only get file name, without prefix
        saClient.download_blob(
            cname="unblurred",
            blob_name=f"{start_date_dag}/{blob}",
            local_file_path=f"{opt.folder}/{blob}",
        )

    print("downloaded files are")
    print(f"cwd is {os.getcwd()}")
    print(f"ls of files {os.listdir(os.getcwd())}")
    print(os.listdir(Path(os.getcwd(), "unblurred", f"{start_date_dag}")))

    blur_imagery(
        opt.weights,
        opt.batch_size,
        opt.img_size,
        opt.conf_thres,
        opt.iou_thres,
    )

    # upload blurred images to storage account
    for file in os.listdir(f"{opt.output_folder}"):
        saClient.upload_blob(
            cname="blurred",
            blob_name=f"{start_date_dag}/{file}",
            local_file_path=f"{opt.output_folder}/{file}",
        )
