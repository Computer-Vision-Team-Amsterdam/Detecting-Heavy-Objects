"""
This module retrieves images from CloudVPS and downloads them locally.
The images are downloaded in the `retrieved_images` folder.
"""
import json
import os
import pickle
import shutil
import socket
from datetime import datetime, timedelta
from multiprocessing import Pool
from pathlib import Path
from typing import Tuple, Union

import requests
from azure.identity import ManagedIdentityCredential
from azure.keyvault.secrets import SecretClient
from requests.auth import HTTPBasicAuth


client_id = os.getenv("USER_ASSIGNED_MANAGED_IDENTITY")
credential = ManagedIdentityCredential(client_id=client_id)

airflow_secrets = json.loads(os.environ["AIRFLOW__SECRETS__BACKEND_KWARGS"])
KVUri = airflow_secrets["vault_url"]

client = SecretClient(vault_url=KVUri, credential=credential)
username_secret = client.get_secret(name="CloudVpsRawPassword")
password_secret = client.get_secret(name="CloudVpsRawPassword")
socket.setdefaulttimeout(100)

BASE_URL = f"https://3206eec333a04cc980799f75a593505a.objectstore.eu/intermediate/"
USERNAME = username_secret.value
PASSWORD = password_secret.value


def split_pano_id(pano_id: str) -> Tuple[str, str]:
    """
    Splits name of the panorama in TMX* and pano*
    """
    id_name = pano_id.split("_")[0]
    index = pano_id.index("_")
    img_name = pano_id[index + 1 :]
    return id_name, img_name


def download_panorama_from_cloudvps(
    date: datetime, panorama_id: str, output_dir: Path = Path("./retrieved_images")
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
            + f"{date.year}/{str(date.month).zfill(2)}/{str(date.day).zfill(2)}/{id_name}/{img_name}.jpg"
        )

        response = requests.get(
            url, stream=True, auth=HTTPBasicAuth(USERNAME, PASSWORD)
        )
        if response.status_code == 404:
            raise FileNotFoundError(f"No resource found at {url}")
        filename = f"./{output_dir}/{panorama_id}.jpg"
        with open(filename, "wb") as out_file:
            shutil.copyfileobj(response.raw, out_file)
        del response

        print(f"{panorama_id} completed.")
    except Exception as e:
        print(f"Failed for panorama {panorama_id}:\n{e}")


if __name__ == "__main__":
    pano_dates = [datetime(2016, 3, 17), datetime(2016, 3, 17), datetime(2016, 3, 17)]
    pano_ids = [
        "TMX7315120208-000020_pano_0000_000000",
        "TMX7315120208-000020_pano_0000_000001",
        "TMX7315120208-000020_pano_0000_000002",
    ]

    for pano_date, pano_id in zip(pano_dates, pano_ids):
        download_panorama_from_cloudvps(pano_date, pano_id)
