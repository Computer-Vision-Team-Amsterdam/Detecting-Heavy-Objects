import json
import os
import socket
from datetime import datetime
import requests

from azure_storage_utils import BaseAzureClient, StorageAzureClient

azClient = BaseAzureClient()
sia_password = azClient.get_secret_value("sia-password-acc")

BASE_URL = "https://acc.api.data.amsterdam.nl/signals/v1/private/signals"

socket.setdefaulttimeout(100)

def to_signal(text: str, date_now, lat_lng: dict):
    return {
        "text": text,
        "location": {
            "geometrie": {
                "type": "Point",
                "coordinates": [lat_lng["lng"], lat_lng["lat"]]
            }
        },
        "category": {
            "sub_category": "signals/v1/public/terms/categories/overlast-in-de-openbare-ruimte/sub_categories/hinderlijk-geplaatst-object"
        },
        "reporter": {
            "email": "cvt@amsterdam.nl"
        },
        "priority": {
            "priority": "low",
        },
        "incident_date_start": date_now.strftime("%Y-%m-%d %H:%M")
    }

def _get_access_token(client_id, client_secret): # TODO change quotes
    token_url = 'https://iam.amsterdam.nl/auth/realms/datapunt-ad-acc/protocol/openid-connect/token'
    payload = {
        'client_id': client_id,
        'client_secret': client_secret,
        'grant_type': 'client_credentials'
    }
    response = requests.post(token_url, data=payload)
    if response.status_code == 200:
        print("The server successfully answered the request.")
        return response.json()["access_token"]
    else:
        response.raise_for_status()

def _get_signals_page(access_token, page):
    if access_token is None:
        raise Exception("Access token cannot be None")

    headers = {'Authorization': "Bearer {}".format(access_token)}
    url = BASE_URL + page
    response = requests.get(url, headers=headers)

    if response.status_code == 200:
        print("The server successfully performed the GET request.")
        return response.json()
    else:
        return response.raise_for_status()

def _post_signal(access_token):
    if access_token is None:
        raise Exception("Access token cannot be None")

    headers = {'Authorization': "Bearer {}".format(access_token)}

    text = "CVT Dit is een automatisch gegenereerd signaal."
    date_now: datetime = datetime.now()
    lat_lng = {"lat": 52.367527, "lng": 4.901257}  # TODO

    response = requests.post(
        BASE_URL,
        json=to_signal(text, date_now, lat_lng),
        headers=headers
    )

    if response.status_code == 200:
        print("The server successfully performed the POST request.")
        return response.json()
    else:
        return response.raise_for_status()

# def image_upload():
    # # TODO get image from blob storage, download it
    # list_endpoint = '/signals/v1/private/signals/'
    # detail_endpoint = list_endpoint + '{}'
    # attachment_endpoint = detail_endpoint + '/attachments/'
    # test_host = 'http://testserver'
    #
    # endpoint = attachment_endpoint.format(self.signal.id)
    # image = SimpleUploadedFile('image.gif', small_gif, content_type='image/gif')
    #
    # response = self.client.post(endpoint, data={'file': image})

    # list_endpoint = '/signals/v1/private/signals/'
    # detail_endpoint = list_endpoint + '{}'
    # attachment_endpoint = detail_endpoint + '/attachments/'


    #
    # files = {'media': open('test.jpg', 'rb')}
    # requests.post(url, files=files)

# def check_sia_connection():
access_token = _get_access_token("sia-cvt", sia_password)
print(_get_signals_page(access_token, "?page_size=1"))

# print(_post_signal(access_token))

# Get access to the Azure Storage account.
azure_connection = StorageAzureClient(secret_key="data-storage-account-url")

# Download files to the WORKDIR of the Docker container.
azure_connection.download_blob("postprocessing-input", "colors.jpeg", "colors.jpeg")

files = {'media': open('colors.jpeg', 'rb')}

signal_id = "11670"

url = BASE_URL + f"/{signal_id}/attachments/"

headers = {'Authorization': "Bearer {}".format(access_token)}

response = requests.post(url, files=files, headers=headers)

if response.status_code == 200:
    print("The server successfully performed the POST request.")
    print(response.json())
else:
    print(response.raise_for_status())