import socket
from datetime import datetime
import requests
import os

from azure_storage_utils import BaseAzureClient, StorageAzureClient

BASE_URL = "https://acc.api.data.amsterdam.nl/signals/v1/private/signals"
API_MAX_UPLOAD_SIZE = 20*1024*1024  # 20MB = 20*1024*1024

socket.setdefaulttimeout(100)

TEXT = "CVT Dit is een automatisch gegenereerd signaal."
TEXT_NOTE = "Dit is een Notitie"

def _to_signal(date_now, lat_lng: dict):
    # TODO Straatnaam
    # TODO signals/v1/public/terms/categories/overlast-in-de-openbare-ruimte/sub_categories/hinderlijk-geplaatst-object

    return {
        "text": TEXT,
        "location": {
            "geometrie": {
                "type": "Point",
                "coordinates": [lat_lng["lng"], lat_lng["lat"]]
            }
        },
        "category": {
            "sub_category": "/signals/v1/public/terms/categories/overlast-in-de-openbare-ruimte/sub_categories/overig-openbare-ruimte"
        },
        "reporter": {
            "email": "cvt@amsterdam.nl"
        },
        "priority": {
            "priority": "low",
        },
        "notes": {
            "text": "string \n dit is een nieuwe regel. of dit \\n even testen."
        },
        "incident_date_start": date_now.strftime("%Y-%m-%d %H:%M")
    }

def _get_access_token(client_id, client_secret):
    token_url = "https://iam.amsterdam.nl/auth/realms/datapunt-ad-acc/protocol/openid-connect/token"
    payload = {
        'client_id': client_id,
        'client_secret': client_secret,
        'grant_type': "client_credentials"
    }
    response = requests.post(token_url, data=payload)
    if response.status_code == 200:
        print("The server successfully answered the request.")
        return response.json()["access_token"]
    else:
        response.raise_for_status()

def _get_signals_page(auth_headers, page="?page_size=1"):
    response = requests.get(BASE_URL + page, headers=auth_headers)

    if response.status_code == 200:
        print("The server successfully performed the GET request.")
        return response.json()
    else:
        return response.raise_for_status()

def _post_signal(auth_headers, json_content):
    response = requests.post(
        BASE_URL,
        json=json_content,
        headers=auth_headers
    )

    if response.status_code == 201:
        print("The server successfully performed the POST request.")
        return response.json()["id"]
    else:
        return response.raise_for_status()

def _image_upload(auth_headers, filename, url):
    if os.path.getsize(filename) > API_MAX_UPLOAD_SIZE:
        msg = f"File can be a maximum of {API_MAX_UPLOAD_SIZE} bytes in size."
        raise Exception(msg)

    files = {"file": (filename, open(filename, "rb"))}

    response = requests.post(url, files=files, headers=auth_headers)

    if response.status_code == 201:
        print("The server successfully performed the POST request.")
        return response.json()
    else:
        return response.raise_for_status()

if __name__ == "__main__":
    sia_password = BaseAzureClient().get_secret_value(secret_key="sia-password-acc")
    access_token = _get_access_token("sia-cvt", sia_password)
    headers = {'Authorization': "Bearer {}".format(access_token)}

    print(_get_signals_page(headers))

    # TODO
    file_to_upload = "colors.jpeg"
    date_now: datetime = datetime.now()
    lat_lng = {"lat": 52.367527, "lng": 4.901257}
    signal_id = _post_signal(headers, _to_signal(date_now, lat_lng))

    # Get access to the Azure Storage account.
    azure_connection = StorageAzureClient(secret_key="data-storage-account-url")
    # Download files to the WORKDIR of the Docker container.
    azure_connection.download_blob("postprocessing-input", file_to_upload, file_to_upload)

    url = BASE_URL + f"/{signal_id}/attachments/"

    response_json = _image_upload(headers, file_to_upload, url)