import json
import os
from typing import List

from azure.core.exceptions import AzureError, ResourceNotFoundError
from azure.identity import ManagedIdentityCredential
from azure.keyvault.secrets import SecretClient
from azure.storage.blob import BlobClient, BlobServiceClient, ContainerClient


class AzureStorageUtils:
    def __init__(self) -> None:
        self.credential = ManagedIdentityCredential(
            client_id=os.getenv("USER_ASSIGNED_MANAGED_IDENTITY")
        )
        self.key_vault_name = json.loads(os.environ["AIRFLOW__SECRETS__BACKEND_KWARGS"])
        self.key_vault_url = self.key_vault_name["vault_url"]
        self.blob_service_client = BlobServiceClient(
            account_url=self.get_secret_client, credential=self.credential
        )

    def list_containers(self) -> List[str]:
        """List the containers in a Storage Account.
        Returns:
            list: A list of all containers in a Storage Account.
        """

        try:
            return self.blob_service_client.list_containers()
        except AzureError as ex:
            print("List containers operation failed")
            raise ex

    def list_container_content(self, cname: str) -> List[str]:
        """List the content of a container.
        Returns:
            list: A list of all blobs in a container.
        Args:
            cname: Name of the Azure Storage Container.
        """

        try:
            container_client = self.blob_service_client.get_container_client(
                container=cname
            )
            return container_client.list_blobs()
        except AzureError as ex:
            print("List blobs operation failed")
            raise ex

    def upload_blob(self, cname: str, blob_name: str, local_file_path: str) -> None:
        """Upload a file to a container in the cloud.
        Args:
            cname: Name of the Azure Storage Container.
            blob_name: Name of the blob in the container.
            local_file_path: Name given to the file saved locally.
        """

        try:
            blob_client = self.blob_service_client.get_blob_client(
                container=cname, blob=blob_name
            )
            with open(local_file_path, "rb") as data:
                blob_client.upload_blob(data)
            print("upload_blob: {} -> {} {}".format(local_file_path, cname, blob_name))
        except ResourceNotFoundError as ex:
            print("Failed to upload blob.")
            raise ex

    def download_blob(self, cname: str, blob_name: str, local_file_path: str) -> None:
        """Download a blob from a container.
        Args:
            cname: Name of the Azure Storage Container.
            blob_name: Name of the blob in the container.
            local_file_path: Name given to the file saved locally.
        """

        try:
            blob_client = self.blob_service_client.get_blob_client(
                container=cname, blob=blob_name
            )
            with open(local_file_path, "wb") as download_file:
                download_file.write(blob_client.download_blob().readall())
            print(
                "download_blob: {} {} -> {}".format(cname, blob_name, local_file_path)
            )
        except ResourceNotFoundError as ex:
            print("Failed to download blob.")
            raise ex

    def get_secret_client(self) -> SecretClient:
        """Get Azure Secret client.
        Returns:
            SecretClient: Azure secret client.
        """

        try:
            return SecretClient(
                vault_url=self.key_vault_url, credential=self.credential
            )
        except Exception as ex:
            print("Failed to initialise Azure secret client.")
            raise ex

    def get_secret(self, secret_name: str) -> str:
        """Retrieve secret from cloud.
        Args:
            secret_name: Name of the secret.
        Returns:
            str: The value of the secret.
        """

        try:
            return self.get_secret_client().get_secret(secret_name).value
        except Exception as ex:
            print("Failed to get {} from Azure key vault.".format(secret_name))
            raise ex
