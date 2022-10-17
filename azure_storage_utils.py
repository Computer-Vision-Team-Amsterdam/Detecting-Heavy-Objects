import json
import os
from abc import ABC
from typing import List, Optional

from azure.core.exceptions import ResourceNotFoundError
from azure.identity import ManagedIdentityCredential
from azure.keyvault.secrets import SecretClient
from azure.storage.blob import BlobClient, BlobServiceClient, ContainerClient


class BaseAzureClient(ABC):
    def __init__(self, secret_account_url: str) -> None:
        self.credential = ManagedIdentityCredential(
            client_id=os.getenv("USER_ASSIGNED_MANAGED_IDENTITY")
        )
        self.key_vault_name = json.loads(os.environ["AIRFLOW__SECRETS__BACKEND_KWARGS"])
        self.key_vault_url = self.key_vault_name["vault_url"]

        self.secret_account_url = secret_account_url
        self.secret_value = self._get_secret(self.secret_account_url)

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

    def _get_secret(self, secret_name: str) -> str:
        """Retrieve secret from cloud.
        Args:
            secret_name: Name of the secret.
        Returns:
            str: The value of the secret.
        """

        try:
            secret: str = self._get_secret_client().get_secret(secret_name).value
            return secret
        except ResourceNotFoundError as ex:
            print("No value found in Azure key vault for key {}".format(secret_name))
            raise ex
        except Exception as ex:
            print("Failed to get {} from Azure key vault.".format(secret_name))
            raise ex


class StorageAzureClient(BaseAzureClient):
    def __init__(self, secret_account_url: str) -> None:
        super().__init__(secret_account_url)

        self.blob_service_client = BlobServiceClient(
            account_url=self.secret_value, credential=self.credential
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


class KeyVaultAzureClient(BaseAzureClient):
    def __init__(self, secret_account_url: str) -> None:
        super().__init__(secret_account_url)

    def get_secret_value(self) -> str:
        """Retrieve secret.
        Returns:
            str: The value of the secret.
        """

        return self.secret_value