import os
import argparse
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
        default="weights/best.pt",
        help="model.pt path(s)",
    )
    parser.add_argument(
        "--data", type=str, default="data/pano.yaml", help="*.data path"
    )
    parser.add_argument(
        "--folder", type=Path, default="unblurred", help="folder with images to blur"
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
    opt = parser.parse_args()
    opt.data = check_file(opt.data)  # check file
    print(opt)

    blur_imagery(
        opt.weights,
        opt.batch_size,
        opt.img_size,
        opt.conf_thres,
        opt.iou_thres,
    )
