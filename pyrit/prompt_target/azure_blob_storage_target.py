# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from azure.storage.blob import ContainerClient
from dotenv import load_dotenv
import os
import sys
import tempfile
import uuid

from pyrit.memory import MemoryInterface
from pyrit.models import ChatMessage
from pyrit.prompt_target import PromptTarget


class AzureBlobStorageTarget(PromptTarget):
    """
    The StorageAccountTarget takes prompts, saves the prompts to a file, and stores them as a blob in a provided
    storage account container.

    Args:
        container_url: URL to the Azure Blob Storage Container.
        sas_token: Blob SAS token required to authenticate blob operations if anonymous access is not enabled.
        memory: MemoryInterface to use for the class. FileMemory by default.

    Raises:
        KeyError: Raises an exception
    """

    def __init__(
        self,
        *,
        container_url: str = None,
        sas_token: str = None,
        memory: MemoryInterface = None,
    ) -> None:

        load_dotenv()
        if container_url is None:
            try:
                self.container_url = os.getenv("AZURE_STORAGE_ACCOUNT_CONTAINER_URL")
            except KeyError:
                raise Exception(
                    """
                    Make sure your .env file is populated with 'AZURE_STORAGE_ACCOUNT_CONTAINER_URL'
                    for this storage account container.
                    """
                )

        if sas_token is None:
            try:
                sas_token = os.getenv("AZURE_STORAGE_ACCOUNT_SAS_TOKEN")
            except KeyError:
                raise Exception(
                    """
                    Make sure your .env file is populated with 'AZURE_STORAGE_ACCOUNT_SAS_TOKEN'
                    for this storage account container
                    """
                )

        # TODO: Modify this so it can be for anonymous access or with SAS_TOKEN
        # If SAS_TOKEN is not None then append it to the end of the URL
        # Otherwise, assume it is anonymous access
        self.container_client = ContainerClient.from_container_url(
            container_url=self.container_url, credential=sas_token
        )

        self.created_blob_urls: list[str] = []

        super().__init__(memory)

    def list_created_blob_urls(self) -> list[str]:
        return self.created_blob_urls

    def list_all_blob_urls(self) -> list[str]:
        blob_names = self.container_client.list_blob_names()

        blob_urls = []
        for name in blob_names:
            blob_urls.append(self.container_url + "/" + name)

        return blob_urls

    def send_prompt(self, normalized_prompt: str, conversation_id: str, normalizer_id: str) -> str:

        # Create temporary file to upload to Storage Account Container URL
        local_file_name = str(uuid.uuid4()) + ".txt"

        with tempfile.TemporaryDirectory() as temp_dir:
            upload_file_path = os.path.join(temp_dir, local_file_name)

            file = open(file=upload_file_path, mode="w")
            file.write(normalized_prompt)
            file.close()

            print("\nUploading to Azure Storage as blob:\n\t" + local_file_name)

            with open(file=upload_file_path, mode="rb") as file_data:
                self.container_client.upload_blob(
                    name=local_file_name, data=file_data, length=sys.getsizeof(upload_file_path)
                )

        # Track created blob URL
        blob_url = self.container_url + "/" + local_file_name
        self.created_blob_urls.append(blob_url)

        # Add prompt to memory
        # TODO: Need a way to record the blob url with the prompt
        self.memory.add_chat_message_to_memory(
            conversation=ChatMessage(role="user", content=normalized_prompt),
            conversation_id=conversation_id,
            normalizer_id=normalizer_id,
        )

        return blob_url
