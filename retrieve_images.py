"""
This module retrieves images from CloudVPS and downloads them locally.
The images are downloaded in the `retrieved_images` folder.
"""

import os
from multiprocessing import Pool
import socket

import requests
from requests.auth import HTTPBasicAuth
import shutil
from datetime import datetime, timedelta
from pathlib import Path
import pickle
from azure.keyvault.secrets import SecretClient
from azure.identity import DefaultAzureCredential

"""
When you run the corresponding container locally, i.e. retrieve-images, authentication to Azure is required first.
For example, with az login.
Then you can run the script within the container.

Alternatively to skip authentication, you can comment out the keyvault part and 
uncomment the USERNAME and PASSWORD retrieval from env variables, lines 45-47. 
Then you rebuild the image and run the container locally with 
docker run --env KV_USERNAME=${KV_USERNAME} --env KV_PASSWORD=${KV_PASSWORD}
-it --entrypoint=/bin/bash epureanudiana/retrieve-images

Eventually, the container will run in DaVe's subscription.
There they should have a managed identity assigned to the cluster running the script.
That managed identity has an aad objectid that you can assign a key vault role to
Then, the DefaultAzureCredential() method should pick up the managed identity assigned to the cluster 
to authenticate against the key vault.
"""


keyVaultName = os.environ["KEY_VAULT_NAME"]
KVUri = f"https://{keyVaultName}.vault.azure.net"

credential = DefaultAzureCredential()
client = SecretClient(vault_url=KVUri, credential=credential)

username_secret = client.get_secret("CloudVpsRawUsername")
password_secret = client.get_secret("CloudVpsRawPassword")

socket.setdefaulttimeout(100)

BASE_URL = f"https://3206eec333a04cc980799f75a593505a.objectstore.eu/intermediate/"
USERNAME = username_secret.value
PASSWORD = password_secret.value

"""
BASE_URL = f"https://3206eec333a04cc980799f75a593505a.objectstore.eu/intermediate/"
USERNAME = os.environ["KV_USERNAME"]
PASSWORD = os.environ["KV_PASSWORD"]
"""

YEAR = 2016


if Path(f"id_to_date.p").exists():
    ID_TO_DATE = pickle.load(open(f"id_to_date.p", "rb" ))
    print("Dates in dict: ", list(ID_TO_DATE.values()))
else:
    ID_TO_DATE = {}


def split_pano_id(pano_id):
    id_name = pano_id.split('_')[0]
    index = pano_id.index("_")
    img_name = pano_id[index + 1:]
    return id_name, img_name


def find_date(pano_id):
    current_date = datetime(YEAR, 1, 1)
    id_name, img_name = split_pano_id(pano_id)

    while True:
        url = BASE_URL + f"{current_date.year}/{str(current_date.month).zfill(2)}/{str(current_date.day).zfill(2)}/{id_name}/{img_name}.jpg"
        response = requests.get(url, stream=True, auth=HTTPBasicAuth(USERNAME, PASSWORD))
        if current_date.year == 2022:
            break
        elif response.status_code == 404:
            current_date += timedelta(days=1)
        else:
            return current_date.year, current_date.day, current_date.month
    print(f"COULD NOT FIND CORRESPONDING DATE FOR PANO ID: {pano_id}")
    return None, None, None


def download_image(pano_id):
    output_dir = Path("./retrieved_images")
    try:
        if Path(f'./{output_dir}/{pano_id}.jpg').exists():
            return
        id_name, img_name = split_pano_id(pano_id)
        if not id_name in list(ID_TO_DATE.keys()):
            year, day, month = find_date(pano_id)
            ID_TO_DATE[id_name] = [year, day, month]
            pickle.dump(ID_TO_DATE, open(f"id_to_date.p", "wb" ))
        else:
            year, day, month = ID_TO_DATE[id_name]
        if day == None:
            raise Exception
        url = BASE_URL + f"{year}/{str(month).zfill(2)}/{str(day).zfill(2)}/{id_name}/{img_name}.jpg"
        print(url)
        response = requests.get(url, stream=True, auth=HTTPBasicAuth(USERNAME, PASSWORD))
        filename = f'./{output_dir}/{pano_id}.jpg'
        with open(filename, 'wb') as out_file:
            shutil.copyfileobj(response.raw, out_file)
        del response

        print(f"{pano_id} completed")
    except Exception as e:
        print(f"Failed for panorama {pano_id}, {e}")


if __name__ == "__main__":

    pano_ids = ["TMX7315120208-000020_pano_0000_000000",
                "TMX7315120208-000020_pano_0000_000001",
                "TMX7315120208-000020_pano_0000_000002"
                ]

    p = Pool(8)
    p.map(download_image, pano_ids)
