# This module retrieves the metadata of the raw images, queries the panorama API and stores the metadata in the
# metadata table

import os
import json
import socket
import argparse

from datetime import datetime
from typing import List, Dict, Union

import psycopg2
from psycopg2 import Error
from psycopg2.extras import execute_values


from panorama.client import PanoramaClient
from azure.identity import ManagedIdentityCredential
from azure.keyvault.secrets import SecretClient

# ============================ CONNECTIONS =========================== #
client_id = os.getenv("USER_ASSIGNED_MANAGED_IDENTITY")
credential = ManagedIdentityCredential(client_id=client_id)

airflow_secrets = json.loads(os.environ["AIRFLOW__SECRETS__BACKEND_KWARGS"])
KVUri = airflow_secrets["vault_url"]


postgres_info = os.environ["AIRFLOW_CONN_POSTGRES_DEFAULT"]

print(f"Values: {postgres_info}")

client = SecretClient(vault_url=KVUri, credential=credential)
username_secret = client.get_secret(name="postgresUsername")
password_secret = client.get_secret(name="postgresPassword-short")
host_secret = client.get_secret(name="postgresHostname")
socket.setdefaulttimeout(100)

USERNAME = username_secret.value
USERNAME = f"{USERNAME}@cvt-weu-psql-o-01-silnc2achvsfi"
PASSWORD = password_secret.value
HOST = host_secret.value
PORT = "5432"
DATABASE = "postprocessing-output"
TABLE = "images"

# ============================ CONNECTIONS =========================== #

KEY_LIST = ["file_name", "camera_location_lat", "camera_location_lon", "heading", "taken_at"]


def get_metadata_from_id(panorama_id: str) -> Dict[str, Union[str, float, datetime]]:
    """

    :param panorama_id:
    :return:
    """
    pano_object = PanoramaClient.get_panorama(panorama_id)
    pano_metadata = {key: None for key in KEY_LIST}

    pano_metadata["file_name"] = pano_object.id
    pano_metadata["camera_location_lat"] = pano_object.geometry.coordinates[1]
    pano_metadata["camera_location_long"] = pano_object.geometry.coordinates[0]
    pano_metadata["heading"] = pano_object.heading
    pano_metadata["taken_at"] = pano_object.timestamp

    return pano_metadata


def save_metadata(panorama_ids: List[str]) -> List[Dict[str, Union[str, float, datetime]]]:
    """
    This method stores certain metadata according to the defintion below
    :param panorama_ids:
    :return:
    """

    metadata_complete = list()
    for pano_id in panorama_ids:
        metadata_pano = get_metadata_from_id(pano_id)
        metadata_complete.append(metadata_pano)

    return metadata_complete


def upload_metadata(metadata: List[Dict[str, Union[str, float, datetime]]], date: str= ""):
    """
    This method upload the list of metadata information to the postgres database
    :return:
    """

    connection = None
    cursor = None

    try:
        # Connect to an existing database
        connection = psycopg2.connect(
            user=USERNAME,
            password=PASSWORD,
            host=HOST,
            port=PORT,
            database=DATABASE,
        )
        cursor = connection.cursor()
        query = f"INSERT INTO {TABLE} ({','.join(KEY_LIST)}) VALUES %s"
        values = [[value for value in item.values()] for item in metadata]
        execute_values(cursor, query, values)
        connection.commit()

    except (Exception, Error) as error:
        print("Error while connecting to PostgreSQL", error)
    finally:
        if connection:
            cursor.close()
            connection.close()
            print("PostgreSQL connection is closed")


# ============= MAIN METHOD ============= #


def metadata_processing():
    """
    Main method to be run in the pipeline dag.
    """
    # parse date argument
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", type=str, help="date to retrieve images")
    opt = parser.parse_args()

    panorama_ids = ["TMX7315120208-000020_pano_0000_000000",
                    "TMX7315120208-000020_pano_0000_000001",
                    "TMX7315120208-000020_pano_0000_000002"]
    metadata = save_metadata(panorama_ids)
    upload_metadata(metadata, date=opt.date)


metadata_processing()
