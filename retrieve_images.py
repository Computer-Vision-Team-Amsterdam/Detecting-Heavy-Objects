"""
This module retrieves images from CloudVPS,
downloads them locally and uploads them to the storage account
The images are downloaded in the `retrieved_images` folder.
"""
import argparse
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Tuple

import requests
from requests.auth import HTTPBasicAuth

from azure_storage_utils import BaseAzureClient, StorageAzureClient

azClient = BaseAzureClient()
BASE_URL = "https://3206eec333a04cc980799f75a593505a.objectstore.eu/intermediate/"
USERNAME = azClient.get_secret_value("CloudVpsRawUsername")
PASSWORD = azClient.get_secret_value("CloudVpsRawPassword")


def split_pano_id(panorama_id: str) -> Tuple[str, str]:
    """
    Splits name of the panorama in TMX* and pano*
    """
    id_name = panorama_id.split("_")[0]
    index = panorama_id.index("_")
    img_name = panorama_id[index + 1 :]

    return id_name, img_name


def download_panorama_from_cloudvps(
    date: datetime, panorama_id: str, output_dir: Path = Path("retrieved_images")
) -> None:
    """
    Downloads panorama from cloudvps to local folder.
    """

    if Path(f"./{output_dir}/{panorama_id}.jpg").exists():
        print(f"Panorama {panorama_id} is already downloaded.")
        return
    id_name, img_name = split_pano_id(panorama_id)

    try:
        url = (
            BASE_URL + f"{date.year}/"
            f"{str(date.month).zfill(2)}/"
            f"{str(date.day).zfill(2)}/"
            f"{id_name}/{img_name}.jpg"
        )

        response = requests.get(
            url, stream=True, auth=HTTPBasicAuth(USERNAME, PASSWORD)
        )
        if response.status_code == 404:
            raise FileNotFoundError(f"No resource found at {url}")

        if response.status_code != 200:
            raise KeyError(f"Status code is {response.status_code}")

        filename = Path(os.getcwd(), output_dir, f"{panorama_id}.jpg")
        with open(filename, "wb") as out_file:
            shutil.copyfileobj(response.raw, out_file)
        del response

        print(f"{panorama_id} completed.")

    except requests.HTTPError as exception:
        print(f"Failed for panorama {panorama_id}:\n{exception}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", type=str, help="date to retrieve images")
    opt = parser.parse_args()

    saClient = StorageAzureClient(secret_key="data-storage-account-url")

    # Download from Cloud
    cname_input = "retrieve-images-input"
    input_files = saClient.list_container_content(
        cname="retrieve-images-input",
        blob_prefix=opt.date,
    )
    print(
        f"Found {len(input_files)} file(s) in container {cname_input} on date {opt.date}."
    )

    pano_ids = []
    for input_file in input_files:
        local_file = input_file.split("/")[1]
        saClient.download_blob(
            cname="retrieve-images-input",
            blob_name=input_file,
            local_file_path=local_file,
        )
        with open(local_file, "r") as f:
            pano_ids = [line.rstrip("\n") for line in f]

    pano_date = datetime.strptime(opt.date, "%Y-%m-%d").date()

    for pano_id in pano_ids:
        download_panorama_from_cloudvps(pano_date, pano_id)

    # Upload to Cloud
    local_file_path = "retrieved_images"
    for file in os.listdir(local_file_path):
        saClient.upload_blob(
            cname="unblurred",
            blob_name=f"{opt.date}/{file}",
            local_file_path=f"retrieved_images/{file}",
        )
