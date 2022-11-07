"""
This module can be used to upload metadata of unblurred images
as well as predictions of the container detection model.
"""

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import psycopg2
from psycopg2._psycopg import connection  # pylint: disable-msg=E0611
from psycopg2._psycopg import cursor  # pylint: disable-msg=E0611
from psycopg2.errors import ConnectionException  # pylint: disable-msg=E0611
from psycopg2.extras import execute_values

from azure_storage_utils import BaseAzureClient, StorageAzureClient

azClient = BaseAzureClient()
USERNAME = azClient.get_secret_value("postgresUsername")
PASSWORD = azClient.get_secret_value("postgresPassword")
HOST = azClient.get_secret_value("postgresHostname")
PORT = "5432"
DATABASE = "container-detection-database"


def connect() -> Tuple[connection, cursor]:
    """
    Connect to the postgres database.
    """
    conn = None
    cur = None

    try:
        # Connect to an existing database
        conn = psycopg2.connect(
            user=f"{USERNAME}@{HOST}",
            password=PASSWORD,
            host=f"{HOST}.postgres.database.azure.com",
            port=PORT,
            database=DATABASE,
        )
        cur = conn.cursor()

    except ConnectionException as error:
        print("Error while connecting to PostgreSQL", error)

    return conn, cur


def get_column_names(table_name: str, cur: cursor) -> List[str]:
    """
    Get names of the columns in database table.

    :param table_name: name of the table in postgres
    :param cur: cursor to parse the table

    return: list of column names, excluding auto-generated primary key
    """

    sql = f"""SELECT * FROM {table_name}"""
    cur.execute(sql)
    cols = [desc[0] for desc in cursor.description]

    # currently all tables' PK have 'id' in them.
    # For 'images' table, PK=file_name, NOT autogenerated
    if "id" in cols[0]:
        del cols[0]

    return cols


def row_to_upload(
    data_element: Dict[str, Union[str, float]],
    object_fields: List[str],
    table_columns: List[str],
) -> Dict[str, Union[str, float]]:
    """
    Creates row with data to upload to table such that
    the object keys match the table columns.

    :param data_element: data to upload in a single row
    :param object_fields: fields from the data object to be considered for upload
    :param table_columns: corresponding columns from table

    :return: object with ordered row-content to be inserted into table
    """
    row: Dict[str, Union[str, float]] = {key: "" for key in table_columns}

    if len(table_columns) != len(row):
        raise ValueError(
            "You are trying to add more/less columns than current table columns."
        )
    for i, column_name in enumerate(table_columns):
        row[column_name] = data_element[object_fields[i]]

    return row


def row_to_upload_from_panorama(
    panorama_id: str, table_columns: List[str]
) -> Dict[str, Union[str, float, datetime]]:
    """
    Creates row with data to upload to table such that the object keys match
    the table columns. Similar to row_to_upload(), but the structure of
    Panorama object makes it difficult to use a for-loop.
    Hence, this is a separate function.

    :param panorama_id: panorama id to query the API with.
    :param table_columns: columns from table corresponding to
                        the selected fields from the panorama object

    :return: object with ordered row-content to be inserted into table
    """
    from panorama.client import PanoramaClient

    pano_object = PanoramaClient.get_panorama(panorama_id)
    row: Dict[str, Union[str, float, datetime]] = {key: "" for key in table_columns}

    row["file_name"] = pano_object.id + ".jpg"
    row["camera_location_lat"] = pano_object.geometry.coordinates[1]
    row["camera_location_lon"] = pano_object.geometry.coordinates[0]
    row["heading"] = pano_object.heading
    row["taken_at"] = pano_object.timestamp

    assert len(row) == len(table_columns)

    return row


def combine_rows_to_upload(
    data: Union[List[str], Union[List[Dict[str, Union[str, float, datetime]]]]],
    object_fields: List[Optional[str]],
    table_columns: List[str],
) -> List[Dict[str, Union[str, float, datetime]]]:
    """
    Creates list of rows with data to upload to table.

    :param data: list of pano ids OR list of dicts with detections from detectron2
    :param object_fields: fields from the data object to be considered for upload
    :param table_columns: corresponding columns from table

    :return: rows to upload to table
    """
    is_object_fields = (
        len(object_fields) != 0
    )  # if there are no object fields passed, we query panorama for element

    rows: List[Dict[str, Union[str, float, datetime]]] = [
        row_to_upload(element, object_fields, table_columns)  # type: ignore
        if is_object_fields
        else row_to_upload_from_panorama(element, table_columns)  # type: ignore
        for element in data
    ]

    return rows


def upload_input(
    conn: connection,
    cur: cursor,
    table_name: str,
    data: Union[List[str], List[Dict[str, Union[str, float, datetime]]]],
    object_fields: List[Union[None, str]],
) -> None:
    """
    Uploads rows to table in postgres.

    :param conn: connection to database
    :param cur: table parser
    :param table_name: name of database table to upload to
    :param data: input data; either a list of panoramas ids
                OR list of dicts with predictions
    :param object_fields: fields from the data object to be considered for upload

    """

    keys = get_column_names(table_name, cur)  # column names from table in postgres
    to_upload_data: List[
        Dict[str, Union[str, float, datetime]]
    ] = combine_rows_to_upload(data, object_fields, table_columns=keys)

    query = f"INSERT INTO {table_name} ({','.join(keys)}) VALUES %s"
    values = [list(item.values()) for item in to_upload_data]

    execute_values(cur, query, values)
    conn.commit()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--table",
        type=str,
        choices=["images", "detections"],
        help="table in postgres where to upload data",
    )
    parser.add_argument(
        "--date",
        type=str,
        help="date when pipeline is run",
    )
    opt = parser.parse_args()

    object_fields_to_select: List[Optional[str]] = []
    if opt.table == "images":
        input_data = [
            "TMX7315120208-000020_pano_0000_000000",
            "TMX7315120208-000020_pano_0000_000001",
            "TMX7315120208-000020_pano_0000_000002",
            "TMX7316010203-001697_pano_0000_000220",
            "TMX7316010203-001697_pano_0000_000215",
            "TMX7316010203-001697_pano_0000_000216",
            "TMX7316010203-001697_pano_0000_000217",
        ]
        object_fields_to_select = []

    if opt.table == "detections":

        # download detections file from the storage account
        saClient = StorageAzureClient(secret_key="data-storage-account-url")

        if not Path(opt.date).exists():
            Path(opt.date).mkdir(exist_ok=True, parents=True)

        input_file_path = f"{opt.date}/coco_instances_results.json"
        saClient.download_blob(
            cname="detections",
            blob_name=f"{opt.date}/coco_instances_results.json",
            local_file_path=f"{opt.date}/coco_instances_results.json",
        )

        f = open(input_file_path)
        input_data = json.load(f)
        object_fields_to_select = ["pano_id", "score", "bbox"]

        # Get all images with detections (with duplicates)
        images_w_det = [item["pano_id"] for item in input_data]

        images_all = saClient.list_container_content(
            cname="blurred",
            blob_prefix=opt.date,
        )

        images_to_remove = set(images_all) - set(images_w_det)
        print(f"Images without a detection: {images_to_remove}")
        saClient.delete_blob(
            cname="blurred",
            blob_names=images_to_remove,
            blob_prefix=opt.date,
        )


    connection, cursor = connect()
    upload_input(connection, cursor, opt.table, input_data, object_fields_to_select)

    if connection:
        cursor.close()
        connection.close()
        print("PostgreSQL connection is closed")
