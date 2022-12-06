import json
import os
from typing import List, Optional

from azure.core.exceptions import ResourceNotFoundError
from azure.identity import ManagedIdentityCredential
from azure.keyvault.secrets import SecretClient
from azure.storage.blob import BlobServiceClient


class BaseAzureClient:
    def __init__(self) -> None:
        self.credential = ManagedIdentityCredential(
            client_id=os.getenv("USER_ASSIGNED_MANAGED_IDENTITY")
        )
        self.key_vault_name = json.loads(os.environ["AIRFLOW__SECRETS__BACKEND_KWARGS"])
        self.key_vault_url = self.key_vault_name["vault_url"]

        self.secret_client = self._get_secret_client()

    def _get_secret_client(self) -> SecretClient:
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

    def get_secret_value(self, secret_key: str) -> str:
        """Retrieve secret from cloud.
        Args:
            secret_key: Name of the secret.
        Returns:
            str: The value of the secret.
        """

        try:
            secret: str = self.secret_client.get_secret(secret_key).value
            return secret
        except ResourceNotFoundError as ex:
            print("No value found in Azure key vault for key {}".format(secret_key))
            raise ex
        except Exception as ex:
            print("Failed to get {} from Azure key vault.".format(secret_key))
            raise ex


class StorageAzureClient(BaseAzureClient):
    def __init__(self, secret_key: str) -> None:
        """
        param secret_key: name of the storage account url. Currently stored in a secret key in the key vault
        """
        super().__init__()

        self.secret_key = secret_key
        self.blob_service_client = BlobServiceClient(
            account_url=self.get_secret_value(self.secret_key),
            credential=self.credential,
        )

    def list_containers(self) -> List[str]:
        """List the containers in a Storage Account.
        Returns:
            list: A list of all containers in a Storage Account.
        """

        try:
            containers = self.blob_service_client.list_containers()
            return [container.name for container in containers]
        except Exception as ex:
            print("List containers operation failed")
            raise ex

    def list_container_content(
        self, cname: str, blob_prefix: Optional[str] = None
    ) -> List[str]:
        """List the content of a container.
        Args:
            cname: Name of the Azure Storage Container.
            blob_prefix: Filters only blobs whose names begin with the specified prefix.
        Returns:
            list: A list of all blobs in a container.
        """

        try:
            container_client = self.blob_service_client.get_container_client(
                container=cname
            )
            blobs = container_client.list_blobs(name_starts_with=blob_prefix)
            return [blob.name for blob in blobs]
        except Exception as ex:
            print("List blobs operation failed")
            raise ex

    def delete_blobs(self, cname: str, blob_names: List[str]) -> None:
        """Delete blobs from a container in the cloud.
        Args:
            cname: Name of the Azure Storage Container.
            blob_names: Names of the blobs you want to delete.
            blob_prefix: The base folder of a blob. For example the date.
        """
        try:
            container_client = self.blob_service_client.get_container_client(
                container=cname
            )
            for blob_name in blob_names:
                container_client.delete_blob(blob_name)
        except Exception as ex:
            print("Delete blobs operation failed")
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
