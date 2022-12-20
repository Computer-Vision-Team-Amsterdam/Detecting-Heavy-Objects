"""
This module retrieves images from CloudVPS,
downloads them locally and uploads them to the storage account
The images are downloaded in the `retrieved_images` folder.
"""
import argparse
import os
import shutil
from datetime import datetime, timedelta
from pathlib import Path
from typing import Tuple
import json
from collections import defaultdict

import requests
from requests.auth import HTTPBasicAuth

from utils.azure_storage import BaseAzureClient, StorageAzureClient
from utils.date import get_start_date

azClient = BaseAzureClient()
BASE_URL = azClient.get_secret_value("CloudVpsBlurredUrl")
USERNAME = azClient.get_secret_value("CloudVpsBlurredUsername")
PASSWORD = azClient.get_secret_value("CloudVpsBlurredPassword")


def split_pano_id(panorama_id: str) -> Tuple[str, str]:
    """
    Splits name of the panorama in TMX* and pano*
    """
    id_name = panorama_id.split("_")[0]
    index = panorama_id.index("_")
    img_name = panorama_id[index + 1 :]

    return id_name, img_name


def download_panorama_from_cloudvps(
    date: datetime, panorama_id: str, output_dir: str
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

def get_pano_ids(start_date_dag_ymd, one_day_later):
    """
    Get panoramic image id's for a user defined bounding box region
    """
    pano_url = f"https://api.data.amsterdam.nl/panorama/panoramas/?srid=28992&timestamp_after={start_date_dag_ymd}&timestamp_before={one_day_later}"

    response = requests.get(pano_url)
    if response.ok:
        pano_data_all = json.loads(response.content)
    else:
        print(response.raise_for_status())
        return []

    pano_ids_dict = defaultdict(list)

    pano_data = pano_data_all['_embedded']['panoramas']

    for item in pano_data:
        pano_id = item['pano_id']
        pano_id_key = pano_id.split("_")[0]
        pano_ids_dict[pano_id_key].append(pano_id)

    # Check for next page with data
    next_page = pano_data_all['_links']['next']['href']

    # Exit the while loop if there is no next page
    while next_page:
        with requests.get(next_page) as response:
            pano_data_all = json.loads(response.content)

        pano_data = pano_data_all['_embedded']['panoramas']

        # Append the panorama id's to the list
        for item in pano_data:
            pano_id = item['pano_id']
            pano_id_key = pano_id.split("_")[0]
            pano_ids_dict[pano_id_key].append(pano_id)

        # Check for next page
        next_page = pano_data_all['_links']['next']['href']

    return pano_ids_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--date", type=str, help="Processing date in the format %Y-%m-%d %H:%M:%S.%f"
    )
    opt = parser.parse_args()

    start_date_dag, start_date_dag_ymd = get_start_date(opt.date)

    saClient = StorageAzureClient(secret_key="data-storage-account-url")

    # List contents of Blob Container
    cname_input = "retrieve-images-input"
    input_files = saClient.list_container_content(
        cname=cname_input,
        blob_prefix=start_date_dag_ymd,
    )
    print(
        f"Found {len(input_files)} file(s) in container {cname_input} on date {start_date_dag_ymd}."
    )

    if len(input_files) > 0:
        # Download txt file(s) with pano ids that we want to download from CloudVPS
        pano_ids = []
        for input_file in input_files:
            local_file = input_file.split("/")[1]  # only get file name, without prefix
            saClient.download_blob(
                cname=cname_input,
                blob_name=input_file,
                local_file_path=local_file,
            )
            with open(local_file, "r") as f:
                pano_ids = [line.rstrip("\n") for line in f]
    else:
        # Get pano ids from API that we want to download from CloudVPS
        my_format_ymd = "%Y-%m-%d"
        start_date = datetime.strptime(start_date_dag_ymd, my_format_ymd)
        end_date = start_date + timedelta(days=1)
        one_day_later = end_date.strftime(my_format_ymd)
        pano_ids_dict = get_pano_ids(start_date_dag_ymd, one_day_later)

        pano_ids = []
        for pano_id_item in pano_ids_dict.keys():
            filename_retrieve = f"{pano_id_item}.txt"
            # All pano ids in a flat list
            pano_ids += pano_ids_dict[pano_id_item]

            with open(filename_retrieve, "w") as f:
                for s in pano_ids_dict[pano_id_item]:
                    f.write(s + "\n")

            saClient.upload_blob(
                cname=cname_input,
                blob_name=f"{start_date_dag_ymd}/{filename_retrieve}",
                local_file_path=filename_retrieve,
            )

    print(
        f"Found {len(pano_ids)} panoramas that will be downloaded from CloudVPS."
    )

    # Download files from CloudVPS
    local_file_path = "retrieved_images"
    for pano_id in pano_ids:
        download_panorama_from_cloudvps(
            datetime.strptime(start_date_dag_ymd, "%Y-%m-%d"), pano_id, local_file_path
        )

    # Upload images to Cloud
    for file in os.listdir(local_file_path):
        saClient.upload_blob(
            cname="unblurred",
            blob_name=f"{start_date_dag}/{file}",
            local_file_path=f"{local_file_path}/{file}",
        )
