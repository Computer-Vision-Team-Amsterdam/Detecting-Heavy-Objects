"""
This module retrieves images from CloudVPS,
downloads them locally and uploads them to the storage account
The images are downloaded in the `retrieved_images` folder.
"""
import json
import os
import argparse
import shutil
import socket
from datetime import datetime
from pathlib import Path
from typing import Tuple

import requests
from azure.identity import ManagedIdentityCredential
from azure.keyvault.secrets import SecretClient
from azure.storage.blob import BlobServiceClient

from requests.auth import HTTPBasicAuth


client_id = os.getenv("USER_ASSIGNED_MANAGED_IDENTITY")
credential = ManagedIdentityCredential(client_id=client_id)
blob_service_client = BlobServiceClient(
    account_url="https://cvtdataweuogidgmnhwma3zq.blob.core.windows.net",
    credential=credential)

airflow_secrets = json.loads(os.environ["AIRFLOW__SECRETS__BACKEND_KWARGS"])
KVUri = airflow_secrets["vault_url"]

client = SecretClient(vault_url=KVUri, credential=credential)
username_secret = client.get_secret(name="CloudVpsRawUsername")
password_secret = client.get_secret(name="CloudVpsRawPassword")
socket.setdefaulttimeout(100)

BASE_URL = "https://3206eec333a04cc980799f75a593505a.objectstore.eu/intermediate/"
USERNAME = username_secret.value
PASSWORD = password_secret.value


def split_pano_id(panorama_id: str) -> Tuple[str, str]:
    """
    Splits name of the panorama in TMX* and pano*
    """
    id_name = panorama_id.split("_")[0]
    index = panorama_id.index("_")
    img_name = panorama_id[index + 1:]
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
                BASE_URL
                + f"{date.year}/"
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
        # filename = f"./{output_dir}/{panorama_id}.jpg"
        filename = Path(os.getcwd(), output_dir, f"{panorama_id}.jpg")
        with open(filename, "wb") as out_file:
            shutil.copyfileobj(response.raw, out_file)
        del response

        print(f"{panorama_id} completed.")

        path = os.path.join(os.getcwd(), "retrieved_images", f"{panorama_id}.jpg")
        size = float(os.path.getsize(path))
        size = size / 1000000
        print(f"size in MB of the downloaded file: {size}")
    except requests.HTTPError as exception:
        print(f"Failed for panorama {panorama_id}:\n{exception}")


def upload_to_storage_account(date: str) -> None:
    """
    This method uploads images in a specific date sub-blob in the storage account
    """
    retrieved_images_folder_path = "retrieved_images"
    for file in os.listdir(retrieved_images_folder_path):
        blob_client = blob_service_client.get_blob_client(
            container="unblurred", blob=f"{date}/{file}")

        # Upload the created file
        with open(os.path.join(retrieved_images_folder_path, file), "rb") as data:
            blob_client.upload_blob(data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", type=str, help="date to retrieve images")
    opt = parser.parse_args()

    pano_dates = [datetime(2016, 3, 17), datetime(2016, 3, 17), datetime(2016, 3, 17)]
    pano_ids = [
        "TMX7315120208-000020_pano_0000_000000",
        "TMX7315120208-000020_pano_0000_000001",
        "TMX7315120208-000020_pano_0000_000002",
    ]

    for pano_date, pano_id in zip(pano_dates, pano_ids):
        download_panorama_from_cloudvps(pano_date, pano_id)

    upload_to_storage_account(opt.date)
