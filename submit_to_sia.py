import json
import os
import socket
from typing import Final
from datetime import datetime
import requests
from azure.identity import ManagedIdentityCredential
from azure.keyvault.secrets import SecretClient

# Command that you want to run on container start
AKS_NAMESPACE: Final = os.getenv("AIRFLOW__KUBERNETES__NAMESPACE")
AKS_NODE_POOL: Final = "cvision2work"

BASE_URL = "https://acc.api.data.amsterdam.nl/signals/v1/private/signals"

client_id = os.getenv("USER_ASSIGNED_MANAGED_IDENTITY")
credential = ManagedIdentityCredential(client_id=client_id)

airflow_secrets = json.loads(os.environ["AIRFLOW__SECRETS__BACKEND_KWARGS"])
KVUri = airflow_secrets["vault_url"]

client = SecretClient(vault_url=KVUri, credential=credential)
sia_password = client.get_secret(name="sia-password-acc")
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
#     # TODO get image from blob storage, download it
    # list_endpoint = '/signals/v1/private/signals/'
    # detail_endpoint = list_endpoint + '{}'
    # attachment_endpoint = detail_endpoint + '/attachments/'
    # test_host = 'http://testserver'
    #
    # endpoint = attachment_endpoint.format(self.signal.id)
    # image = SimpleUploadedFile('image.gif', small_gif, content_type='image/gif')
    #
    # response = self.client.post(endpoint, data={'file': image})
    #
    # import requests
    # url = 'http://file.api.wechat.com/cgi-bin/media/upload?access_token=ACCESS_TOKEN&type=TYPE'
    # files = {'media': open('test.jpg', 'rb')}
    # requests.post(url, files=files)

# def check_sia_connection():
access_token = _get_access_token("sia-cvt", f"{sia_password.value}")
# print(_get_signals_page(access_token, "?page_size=1"))

# print(_post_signal(access_token))