"""
This module stores the output file from postprocessing.py, i.e. prioritized_objects in
our development SA. It also copies the output file to the 'containers' table in Postgres.
"""

import os
import shutil
from datetime import datetime
from pathlib import Path

import psycopg2
from azure.storage.blob import BlobServiceClient
from psycopg2 import Error

# ========== CONNECTION TO POSTGRES DATABASE =============

connection = None
try:
    # Connect to an existing database
    connection = psycopg2.connect(
        user=os.environ["USERNAME"],
        password=os.environ["PASSWORD"],
        host=os.environ["HOST"],
        port="5432",
        database="postprocessing-output",
    )

    # Create a cursor to perform database operations
    cursor = connection.cursor()
    # Print PostgreSQL details
    print("PostgreSQL server information")

    # Executing a SQL query
    with open("prioritized_objects.csv", "r") as f:
        next(f)  # Skip the header row.
        cursor.copy_from(f, "containers", sep=";")

    connection.commit()

except (Exception, Error) as error:
    print("Error while connecting to PostgreSQL", error)
finally:
    if connection:
        cursor.close()
        connection.close()
        print("PostgreSQL connection is closed")

# ============= CONNECTION TO STORAGE ACCOUNT ==========

try:

    connect_str = os.environ["AZURE_STORAGE_CONNECTION_STRING"]
    blob_service_client: BlobServiceClient = BlobServiceClient.from_connection_string(
        connect_str
    )

    today = datetime.today().strftime("%Y-%m-%d")
    local_file_path = "prioritized_objects.csv"

    blob_client = blob_service_client.get_blob_client(
        container="postprocessing-output", blob=f"{today}/{local_file_path}"
    )

    # Upload the created file
    with open(local_file_path, "rb") as data:
        blob_client.upload_blob(data)


except Exception as ex:
    print("Exception:")
    print(ex)
