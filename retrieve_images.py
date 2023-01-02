"""
This module retrieves images from CloudVPS,
downloads them locally and uploads them to the storage account
The images are downloaded in the `retrieved_images` folder.
"""
import argparse
import json
import os
import shutil
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, List, Tuple

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
    id_name, pano_name = split_pano_id(panorama_id)

    try:
        url = (
            BASE_URL + f"{date.year}/"
            f"{str(date.month).zfill(2)}/"
            f"{str(date.day).zfill(2)}/"
            f"{id_name}/{pano_name}/"
            f"equirectangular/panorama_8000.jpg"
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


def get_pano_ids(start_date_dag_ymd: str) -> Any:
    """
    Get panoramic image id's for a user defined bounding box region
    """
    my_format_ymd = "%Y-%m-%d"
    start_date = datetime.strptime(start_date_dag_ymd, my_format_ymd)
    end_date = start_date + timedelta(days=1)
    one_day_later = end_date.strftime(my_format_ymd)

    pano_url = (
        f"https://api.data.amsterdam.nl/panorama/panoramas/?srid=28992&timestamp_after={start_date_dag_ymd}"
        f"&timestamp_before={one_day_later}"
    )

    response = requests.get(pano_url)
    if response.ok:
        pano_data_all = json.loads(response.content)
    else:
        response.raise_for_status()

    pano_ids_dict = defaultdict(list)

    pano_data = pano_data_all["_embedded"]["panoramas"]

    for item in pano_data:
        pano_id = item["pano_id"]
        pano_id_key = pano_id.split("_")[0]
        pano_ids_dict[pano_id_key].append(pano_id)

    # Check for next page with data
    next_page = pano_data_all["_links"]["next"]["href"]

    # Exit the while loop if there is no next page
    while next_page:
        with requests.get(next_page) as response:
            pano_data_all = json.loads(response.content)

        pano_data = pano_data_all["_embedded"]["panoramas"]

        # Append the panorama id's to the list
        for item in pano_data:
            pano_id = item["pano_id"]
            pano_id_key = pano_id.split("_")[0]
            pano_ids_dict[pano_id_key].append(pano_id)

        # Check for next page
        next_page = pano_data_all["_links"]["next"]["href"]

    return pano_ids_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--date", type=str, help="Processing date in the format %Y-%m-%d %H:%M:%S.%f"
    )
    parser.add_argument("--num-workers", type=int, help="number of workers")
    opt = parser.parse_args()

    start_date_dag, start_date_dag_ymd = get_start_date(opt.date)

    saClient = StorageAzureClient(secret_key="data-storage-account-url")

    development = True
    if development:
        # TODO only works for date {"date":"2020-05-08 00:00:00.00"}
        pano_ids = [
            "TMX7316010203-001697_pano_0000_000170",
            "TMX7316010203-001697_pano_0000_000190",
            "TMX7316010203-001697_pano_0000_000200",
            "TMX7316010203-001697_pano_0000_000220",
            "TMX7316010203-001697_pano_0000_000215",
            "TMX7316010203-001697_pano_0000_000216",
            "TMX7316010203-001697_pano_0000_000217",
        ]
    else:
        # Get pano ids from API that we want to download from CloudVPS
        pano_ids_dict = get_pano_ids(start_date_dag_ymd)
        # Pano ids to a flat list (you can also exclude pano keys in pano_ids_dict.keys()).
        pano_ids = []
        for pano_id_item in pano_ids_dict.keys():
            pano_ids += pano_ids_dict[pano_id_item]

    # # TODO Check if pano ids are already processed today
    # List virtual directories in Azure Blob Storage
    # all_dirs = saClient.list_container_directories(cname="retrieve-images-input")
    # start_date = datetime.strptime(opt.date, "%Y-%m-%d %H:%M:%S.%f")
    # pano_ids_processed = []
    # for dir_name in all_dirs:
    #     # Convert string to datatime object
    #     dir_date = datetime.strptime(dir_name, "%Y-%m-%d_%H-%M-%S")
    #     # Compare dir name with the start DAG date
    #     if dir_date < start_date:
    #         print("TODO get all files inside this folder along with the contents (pano ids)")
    #         # TODO do pano_ids_processed.extend(the_pano_ids_list)
    # TODO pano_ids = set(pano_ids) - set(pano_ids_processed)

    print(f"Found {len(pano_ids)} panoramas that will be downloaded from CloudVPS.")

    # TODO split data "pano_ids" in chunks and save 1.txt, 2.txt etc to retrieve-images/{start_date_dag}

    if len(pano_ids) < opt.num_workers:
        raise ValueError(
            "Number of workers is larger than items to process. Aborting..."
        )

    workers = list(range(opt.num_workers))
    split_list = [[] for x in workers]  # type: List[List[str]]
    for i, x in enumerate(pano_ids):
        split_list[i % len(workers)].append(x)

    for chunk_id, chunk_data in enumerate(split_list, 1):
        filename_chunk = f"{chunk_id}.txt"

        with open(filename_chunk, "w") as f:
            for s in chunk_data:
                f.write(s + "\n")

        saClient.upload_blob(
            cname="retrieve-images-input",
            blob_name=f"{start_date_dag}/{filename_chunk}",
            local_file_path=filename_chunk,
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
