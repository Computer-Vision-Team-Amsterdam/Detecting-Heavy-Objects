# download blurred images from cloudvps
import csv
import os
from urllib.error import HTTPError
from urllib.request import urlretrieve
import time
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


keyVaultName = os.environ["KEY_VAULT_NAME"]
KVUri = f"https://{keyVaultName}.vault.azure.net"

print("a")
credential = DefaultAzureCredential()
client = SecretClient(vault_url=KVUri, credential=credential)

print("aa")

username_secret = client.get_secret("CloudVpsRawUsername")
password_secret = client.get_secret("CloudVpsRawPassword")
print("aaa")

socket.setdefaulttimeout(100)

BASE_URL = f"https://3206eec333a04cc980799f75a593505a.objectstore.eu/processed/"
USERNAME = username_secret.value
PASSWORD = password_secret.value


YEAR = 2018


if Path(f"id_to_date.p").exists():
    ID_TO_DATE = pickle.load(open(f"id_to_date.p", "rb" ))
    print("Dates in dict: ", list(ID_TO_DATE.values()))
else:
    ID_TO_DATE = {}
print("aaaa")


def split_pano_id(pano_id):
    id_name = pano_id.split('_')[0]
    index = pano_id.index("_")
    img_name = pano_id[index + 1:]
    return id_name, img_name


def find_date(pano_id):
    current_date = datetime(YEAR, 1, 1)
    id_name, img_name = split_pano_id(pano_id)

    while True:
        url = BASE_URL + f"{current_date.year}/{str(current_date.month).zfill(2)}/{str(current_date.day).zfill(2)}/{id_name}/{img_name}/equirectangular/panorama_4000.jpg"
        response = requests.get(url, stream=True, auth=HTTPBasicAuth(USERNAME, PASSWORD))
        print(url)
        if current_date.year == 2020:
            break
        elif response.status_code == 404:
            current_date += timedelta(days=1)
        else:
            return current_date.year, current_date.day, current_date.month
    print(f"COULD NOT FIND CORRESPONDING DATE FOR PANO ID: {pano_id}")
    return None, None, None


def download_image(pano_id):
    output_dir = Path("./blurred_panos_from_cloudvps")
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
        url = BASE_URL + f"{year}/{str(month).zfill(2)}/{str(day).zfill(2)}/{id_name}/{img_name}/equirectangular/panorama_4000.jpg"
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

    BATCH_START = 100
    BATCH_END = 101

    for i in range(BATCH_START, BATCH_END):
        with open(f'../../data_azure/pano_ids/{i}.txt') as f:
            pano_ids = f.read().splitlines()
        f.close()


    p = Pool(8)
    p.map(download_image, pano_ids)


