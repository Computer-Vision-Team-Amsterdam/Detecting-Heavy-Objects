import argparse
import os
from pathlib import Path
from typing import Union
import cv2

import glob
from tqdm import tqdm


def blur(image: str, label: Union[Path, str], output_path):
    img = cv2.imread(image)
    if img is None:
        print(image)
    dh, dw, _ = img.shape

    fl = open(label, 'r')
    data = fl.readlines()
    fl.close()

    for dt in data:

        _, x, y, w, h = map(float, dt.split(' '))

        l = int((x - w / 2) * dw)
        r = int((x + w / 2) * dw)
        t = int((y - h / 2) * dh)
        b = int((y + h / 2) * dh)

        if l < 0:
            l = 0
        if r > dw - 1:
            r = dw - 1
        if t < 0:
            t = 0
        if b > dh - 1:
            b = dh - 1

        blurred_patch = cv2.GaussianBlur(img[t:b, l:r, :], (51, 51), 0)
        blurred_patch = cv2.GaussianBlur(blurred_patch, (71, 71), 0)

        img[t:b, l:r, :] = blurred_patch

    cv2.imwrite(os.path.join(output_path, f"{Path(image).stem}.jpg"), img)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default="data_azure/17mar2021", help='path to root folder')
    parser.add_argument("--source", type=int, default=1, help="path to batch folder")
    args = parser.parse_args()

    Path(f"{args.root}", f"{args.source}/blurred").mkdir(parents=True, exist_ok=True)
    target_dir = Path(f"{args.root}", f"{args.source}/blurred")

    panorama_ids = [Path(p).stem for p in glob.glob(f"{args.root}/{args.source}/images/*.jpg")]
    for panorama_id in tqdm(panorama_ids, desc="Blurring panoramas"):
        label = f"{args.root}/{args.source}/images/{panorama_id}.txt"
        image = f"{args.root}/{args.source}/images/{panorama_id}.jpg"
        try:
            blur(image, label, target_dir)
        except FileNotFoundError:
            continue


