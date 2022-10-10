import os
import json

from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient
from azure.core.exceptions import ResourceExistsError, ResourceNotFoundError
from azure.identity import ManagedIdentityCredential
from azure.keyvault.secrets import SecretClient

class AzureStorageUtils(object):

    def __init__(self):
        self.credential = ManagedIdentityCredential(client_id=os.getenv("USER_ASSIGNED_MANAGED_IDENTITY"))
        self.key_vault_name = json.loads(os.environ["AIRFLOW__SECRETS__BACKEND_KWARGS"]) # TODO use os.getenv
        self.key_vault_url = self.key_vault_name["vault_url"]
        self.blob_service_client = BlobServiceClient(account_url=self.get_secret_client,
                                                     credential=self.credential)

    def list_containers(self):
        """List the containers in a Storage Account.
        Returns:
            TODO
        """

        try:
            return self.blob_service_client.list_containers()
        except ResourceExistsError:
            return list()

    def list_container_content(self, cname): # TODO
        """List the content of a container.
        Returns:
            TODO
        """

        try:
            container_client = self.blob_service_client.get_container_client(
                container=cname
            )
            return container_client.list_blobs()
        except ResourceExistsError:
            return list()

    def upload_blob(self, local_file_path, cname, blob_name) -> None:
        """Upload a file to a container in the cloud.
        Raises:
            Exception: Exception that will be raised if the operation fails.
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

    def download_blob(self, cname, blob_name, local_file_path) -> None:
        """Download a blob from a container.
        Raises:
            Exception: Exception that will be raised if the operation fails.
        """

        try:
            blob_client = self.blob_service_client.get_blob_client(
                container=cname, blob=blob_name
            )
            with open(local_file_path, "wb") as download_file:
                download_file.write(blob_client.download_blob().readall())
            print("download_blob: {} {} -> {}".format(cname, blob_name, local_file_path))
        except ResourceNotFoundError as ex:
            print("Failed to download blob.")
            raise ex

    def download_blobs(self, cname, cloud_files) -> None:
        # TODO optimize this function?
        """Download certain blobs from a container.
        Raises:
            Exception: Exception that will be raised if the operation fails.
        """

        try:
            container_client = self.blob_service_client.get_container_client(
                container=cname
            )
            blob_list = container_client.list_blobs()

            for blob in blob_list:
                if blob.name in cloud_files:
                    with open(blob.name, "wb") as download_file:
                        download_file.write(container_client.get_blob_client(blob).download_blob().readall())
                    print("download_blob: {} {}".format(cname, blob.name))
        except ResourceNotFoundError as ex:
            print("Failed to download blob.")
            raise ex

    def get_secret_client(self) -> SecretClient:
        """Get Azure Secret client.
        Returns:
            SecretClient: Azure secret client.
        Raises:
            Exception: Exception that will be raised if the operation fails.
        """

        try:
            return SecretClient(
                vault_url=self.key_vault_url,
                credential=self.credential
            )
        except Exception as ex:
            print("Failed to initialise Azure secret client.")
            raise ex

    def get_secret(self, secret_name: str) -> str:
        """Retrieve secret from cloud.
        Args:
            secret_name (str): Name of the secret.
        Returns:
            str: The value of the secret.
        Raises:
            Exception: Exception that will be raised if the operation fails.
        """

        try:
            return self.get_secret_client().get_secret(secret_name).value
        except Exception as ex:
            print(
                "Failed to get {} from Azure key vault.".format(secret_name)
            )
            raise ex