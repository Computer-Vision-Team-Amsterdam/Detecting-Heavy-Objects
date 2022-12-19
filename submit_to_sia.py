import os
import socket
from typing import Any, Dict, List, Optional
import sys

import requests

socket.setdefaulttimeout(100)
import argparse
import json

import pandas.io.sql as sqlio

import upload_to_postgres
from utils.azure_storage import BaseAzureClient, StorageAzureClient
from utils.date import get_start_date

BASE_URL = "https://acc.api.meldingen.amsterdam.nl/signals/v1/private/signals"
API_MAX_UPLOAD_SIZE = 20 * 1024 * 1024  # 20MB = 20*1024*1024

TEXT = (
    "Dit is een automatisch gegenereerd signaal: Met behulp van beeldherkenning is een bouwcontainer of bouwkeet "
    "gedetecteerd op onderstaande locatie, waar waarschijnlijk geen vergunning voor is. N.B. Het adres betreft een "
    "schatting van het dichtstbijzijnde adres bij de containerlocatie, er is geen informatie bekend in hoeverre dit "
    "het adres is van de containereigenaar."
)
DESCRIPTION_ASC = (
    "Instructie ASC:\n"
    "(i) Foto bekijken en alleen signalen doorzetten naar THOR indien er inderdaad een "
    "bouwcontainer of bouwkeet op de foto staat. \n "
    "(ii) De urgentie voor dit signaal moet 'laag' blijven, zodat BOA's dit "
    "signaal herkennen in City Control onder 'Signalering'."
)
DESCRIPTION_BOA = (
    "Instructie BOA’s:\n "
    "(i) Foto bekijken en beoordelen of dit een bouwcontainer of bouwkeet is waar vergunningsonderzoek "
    "ter plaatse nodig is.\n"
    "(ii) Check Decos op aanwezige vergunning voor deze locatie of vraag de vergunning op bij "
    "containereigenaar.\n "
    "(iii) Indien geen geldige vergunning, volg dan het reguliere handhavingsproces."
)

MAX_SIGNALS_TO_SEND = 10
MAX_BUILDING_SEARCH_RADIUS = 50


def _get_description(permit_distance: str, bridge_distance: str) -> str:
    return (
        f"Categorie Rood: 'mogelijk illegaal object op kwetsbare kade'\n"
        f"Afstand tot kwetsbare kade: {bridge_distance} meter\n"
        f"Afstand tot objectvergunning: {permit_distance} meter"
    )


def _to_signal(start_date_dag: str, lat_lon: Dict[str, float], bag_data: List[Any]) -> Any:
    json_to_send = {
        "text": TEXT,
        "location": {
            "geometrie": {
                "type": "Point",
                "coordinates": [lat_lon["lon"], lat_lon["lat"]],
            },
        },
        "category": {
            "sub_category": "/signals/v1/public/terms/categories/overlast-in-de-openbare-ruimte/"
            "sub_categories/hinderlijk-geplaatst-object"
        },
        "reporter": {"email": "cvt@amsterdam.nl"},
        "priority": {
            "priority": "low",
        },
        "incident_date_start": start_date_dag,
    }

    if bag_data:
        location_json = {
            "location": {
                "geometrie": {
                    "type": "Point",
                    "coordinates": [lat_lon["lon"], lat_lon["lat"]],
                },
                "address": {
                    "openbare_ruimte": bag_data[0],
                    "huisnummer": bag_data[1],
                    "postcode": bag_data[2],
                    "woonplaats": "Amsterdam",
                },
            }
        }

        json_to_send.update(location_json)

    return json_to_send


def _get_bag_address_in_range(location_point: Dict[str, float]) -> List[Optional[str]]:
    """
    For a location point, get the nearest building information.
    """
    bag_url = (
        f"https://api.data.amsterdam.nl/bag/v1.1/nummeraanduiding/"
        f"?format=json&locatie={location_point['lat']},{location_point['lon']},"
        f"{MAX_BUILDING_SEARCH_RADIUS}&srid=4326&detailed=1"
    )

    response = requests.get(bag_url)
    if response.status_code == 200:
        response_content = json.loads(response.content)
        if response_content["count"] > 0:
            # Get first element
            first_element = json.loads(response.content)["results"][0]
            return [first_element["openbare_ruimte"]["_display"], first_element["huisnummer"],
                    first_element["postcode"]]
        else:
            print(f"No BAG address in the range of {MAX_BUILDING_SEARCH_RADIUS} found.")
            return []
    else:
        print(f"Failed to get address from BAG, status code {response.status_code}.")
        return []


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
    # Example input: {"date":"2018-05-03 00:00:00.00","container-id-list":"1,2,3,4"}
    parser = argparse.ArgumentParser(
        description="Run postprocessing for container detection pipeline"
    )
    parser.add_argument(
        "--date",
        type=str,
        help="Processing date in the format %Y-%m-%d %H:%M:%S.%f",
    )
    parser.add_argument(
        "--container-id-list",
        type=str,
        help="Delimited list input of ids corresponding to the 'containers' database table.",
    )
    args = parser.parse_args()

    add_notification = False  # TODO make this an argument
    manual_filter = True  # TODO make this an argument

    print(f"List of ranking IDs as input: {args.container_id_list}")
    ids_to_send = [int(item) for item in args.container_id_list.split(",")]
    print(f"List of ranking IDs as input (parsed to ints): {ids_to_send}")

    start_date_dag, start_date_dag_ymd = get_start_date(args.date)

    # Get API data
    sia_password = BaseAzureClient().get_secret_value(secret_key="sia-password-acc")
    access_token = _get_access_token("sia-cvt", sia_password)
    headers = {"Authorization": "Bearer {}".format(access_token)}
    # Get access to the Azure Storage account.
    azure_connection = StorageAzureClient(secret_key="data-storage-account-url")

    # Get images with a detection
    sql = (
        f"SELECT * FROM containers A LEFT JOIN images B ON A.closest_image = B.file_name "
        f"WHERE date_trunc('day', B.taken_at) = '{start_date_dag_ymd}'::date AND A.score <> 0 ORDER "
        f"BY A.score DESC LIMIT '{MAX_SIGNALS_TO_SEND}';"
    )

    with upload_to_postgres.connect() as (conn, _):
        query_df = sqlio.read_sql_query(sql, conn)

    if query_df.empty:
        print(
            "DataFrame is empty! No illegal containers are found for the provided date. Aborting..."
        )
        sys.exit()

    if manual_filter:
        # Select Pandas rows based on list index
        query_df = query_df.iloc[ids_to_send]

    for index, row in query_df.iterrows():
        # Get panoramic image closest to the found container
        closest_image = row["closest_image"]
        # Download files to the WORKDIR of the Docker container
        azure_connection.download_blob(
            cname="blurred",
            blob_name=f"{start_date_dag}/{closest_image}",
            local_file_path=closest_image,
        )

        lat_lon = {"lat": row["lat"], "lon": row["lon"]}

        if add_notification:
            # Get closest building
            address_data = _get_bag_address_in_range(lat_lon)
            # Add a new signal to meldingen.amsterdam.nl
            signal_id = _post_signal(
                headers, _to_signal(start_date_dag, lat_lon, address_data)
            )
            # Add an attachment to the previously created signal
            _image_upload(headers, closest_image, signal_id)

            # Add a description to the previously created signal
            # Description 3
            _patch_signal(
                headers,
                signal_id,
                DESCRIPTION_BOA,
            )
            # Description 2
            _patch_signal(
                headers,
                signal_id,
                DESCRIPTION_ASC,
            )
            # Description 1
            _patch_signal(
                headers,
                signal_id,
                _get_description(row["permit_distance"], row["bridge_distance"]),
            )
