import os
import socket
from datetime import date, datetime
from typing import Any, Dict

import requests

socket.setdefaulttimeout(100)
import argparse

import pandas.io.sql as sqlio

import upload_to_postgres
from azure_storage_utils import BaseAzureClient, StorageAzureClient

BASE_URL = "https://acc.api.meldingen.amsterdam.nl/signals/v1/private/signals"
API_MAX_UPLOAD_SIZE = 20 * 1024 * 1024  # 20MB = 20*1024*1024

TEXT = (
    "Dit is een automatisch gegenereerd signaal. Met behulp van beeldherkenning is een bouwcontainer of bouwkeet "
    "gedetecteerd op onderstaande locatie, waar mogelijk geen vergunning voor is. "
)


def _get_description(permit_distance: str, bridge_distance: str) -> str:
    return (
        f"Categorie Rood: 'mogelijk illegaal object op kwetsbare kade'\n"
        f"Afstand tot kwetsbare kade: {bridge_distance} meter\n"
        f"Afstand tot objectvergunning: {permit_distance} meter\n\n"
        f"Instructie ASC:\n"
        f"o Foto bekijken en alleen signalen doorzetten naar THOR indien er inderdaad een bouwcontainer of "
        f"bouwkeet op de foto staat. \n "
        f"o De urgentie voor dit signaal moet 'laag' blijven, zodat BOA's dit "
        f"signaal herkennen in City Control onder 'Signalering'.\n\n"
        f"Instructie BOA’s:\n "
        f"o Foto bekijken en beoordelen of dit een bouwcontainer of bouwkeet is waar vergunningsonderzoek ter "
        f"plaatse nodig is.\n"
        f"o Check Decos op aanwezige vergunning voor deze locatie of vraag de vergunning op bij containereigenaar.\n "
        f"o Indien geen geldige vergunning, volg dan het reguliere handhavingsproces."
    )


def _to_signal(date_now: date, lat_lng: Dict[str, float]) -> Any:
    return {
        "text": TEXT,
        "location": {
            "geometrie": {
                "type": "Point",
                "coordinates": [lat_lng["lng"], lat_lng["lat"]],
            }
        },
        "category": {
            "sub_category": "/signals/v1/public/terms/categories/overlast-in-de-openbare-ruimte/sub_categories/hinderlijk-geplaatst-object"
        },
        "reporter": {"email": "cvt@amsterdam.nl"},
        "priority": {
            "priority": "low",
        },
        "incident_date_start": date_now.strftime("%Y-%m-%d %H:%M"),
    }


def _get_access_token(client_id: str, client_secret: str) -> Any:
    token_url = "https://iam.amsterdam.nl/auth/realms/datapunt-ad-acc/protocol/openid-connect/token"
    payload = {
        "client_id": client_id,
        "client_secret": client_secret,
        "grant_type": "client_credentials",
    }
    response = requests.post(token_url, data=payload)
    if response.status_code == 200:
        print("The server successfully answered the request.")
        return response.json()["access_token"]
    else:
        return response.raise_for_status()


def _get_signals_page(auth_headers: Dict[str, str], page: str = "?page_size=1") -> Any:
    response = requests.get(BASE_URL + page, headers=auth_headers)

    if response.status_code == 200:
        print("The server successfully performed the GET request.")
        return response.json()
    else:
        return response.raise_for_status()


def _post_signal(auth_headers: Dict[str, str], json_content: Any) -> Any:
    response = requests.post(BASE_URL, json=json_content, headers=auth_headers)

    if response.status_code == 201:
        print("The server successfully performed the POST request.")
        return response.json()["id"]
    else:
        return response.raise_for_status()


def _patch_signal(auth_headers: Dict[str, str], sig_id: str, text_note: str) -> Any:
    json_content = {"notes": [{"text": text_note}]}

    response = requests.patch(
        BASE_URL + f"/{sig_id}", json=json_content, headers=auth_headers
    )

    if response.status_code == 200:
        print("The server successfully performed the POST request.")
        return response.json()
    else:
        return response.raise_for_status()


def _image_upload(auth_headers: Dict[str, str], filename: str, sig_id: str) -> Any:
    if os.path.getsize(filename) > API_MAX_UPLOAD_SIZE:
        msg = f"File can be a maximum of {API_MAX_UPLOAD_SIZE} bytes in size."
        raise Exception(msg)

    files = {"file": (filename, open(filename, "rb"))}

    response = requests.post(
        BASE_URL + f"/{sig_id}/attachments/", files=files, headers=auth_headers
    )

    if response.status_code == 201:
        print("The server successfully performed the POST request.")
        return response.json()
    else:
        return response.raise_for_status()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run postprocessing for container detection pipeline"
    )
    parser.add_argument(
        "--date",
        type=str,
        help="Processing date in the format YYYY-MM-DD",
    )
    args = parser.parse_args()

    add_notification = False  # TODO make this an argument

    # Get API data
    sia_password = BaseAzureClient().get_secret_value(secret_key="sia-password-acc")
    access_token = _get_access_token("sia-cvt", sia_password)
    headers = {"Authorization": "Bearer {}".format(access_token)}
    # Get access to the Azure Storage account.
    azure_connection = StorageAzureClient(secret_key="data-storage-account-url")

    # Make a connection to the database
    conn, cur = upload_to_postgres.connect()

    # Get images with a detection
    sql = f"SELECT * FROM containers;"
    query_df = sqlio.read_sql_query(sql, conn)
    if query_df.empty:
        print(
            "DataFrame is empty! No illegal containers are found for the provided date."
        )

    for index, row in query_df.iterrows():
        # Convert string to datetime object
        date_now = datetime.strptime(args.date, "%Y-%m-%d").date()

        # Get panoramic image closest to the found container
        closest_image = row["closest_image"]
        # Download files to the WORKDIR of the Docker container
        azure_connection.download_blob(
            "blurred", os.path.join(args.date, closest_image), closest_image
        )

        lat_lng = {"lat": row["lat"], "lng": row["lon"]}

        if add_notification:
            # Add a new signal to meldingen.amsterdam.nl
            signal_id = _post_signal(headers, _to_signal(date_now, lat_lng))
            # Add an attachment to the previously created signal
            _image_upload(headers, closest_image, signal_id)
            # Add a description to the previously created signal
            _patch_signal(
                headers,
                signal_id,
                _get_description(row["permit_distance"], row["bridge_distance"]),
            )
