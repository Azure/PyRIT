# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

# TODO: import from azure.storage.blob.aio for async ops
from azure.storage.blob import ContainerClient
import os
import sys
import tempfile
import uuid

from pyrit.common import default_values
from pyrit.memory import MemoryInterface
from pyrit.models import ChatMessage
from pyrit.prompt_target import PromptTarget


class AzureBlobStorageTarget(PromptTarget):
    """
    The StorageAccountTarget takes prompts, saves the prompts to a file, and stores them as a blob in a provided
    storage account container.

    Args:
        container_url: URL to the Azure Blob Storage Container.
        sas_token: Blob SAS token required to authenticate blob operations.
        memory: MemoryInterface to use for the class. FileMemory by default.
    """

    AZURE_STORAGE_CONTAINER_ENVIRONMENT_VARIABLE: str = "AZURE_STORAGE_ACCOUNT_CONTAINER_URL"
    SAS_TOKEN_ENVIRONMENT_VARIABLE: str = "AZURE_STORAGE_ACCOUNT_SAS_TOKEN"

    def __init__(
        self,
        *,
        container_url: str = None,
        sas_token: str = None,
        memory: MemoryInterface = None,
    ) -> None:
        if container_url is None:
            self._container_url = default_values.get_required_value(
                env_var_name=self.AZURE_STORAGE_CONTAINER_ENVIRONMENT_VARIABLE, passed_value=container_url
            )

        if sas_token is None:
            sas_token = default_values.get_required_value(
                env_var_name=self.SAS_TOKEN_ENVIRONMENT_VARIABLE, passed_value=sas_token
            )

        self._client = ContainerClient.from_container_url(
            container_url=self._container_url, credential=sas_token
        )

        self._created_blob_urls: list[str] = []

        super().__init__(memory)

    def list_created_blob_urls(self) -> list[str]:
        return self._created_blob_urls

    def list_all_blob_urls(self) -> list[str]:
        # TODO: When moving this to be async, this is no longer something we can iterate through...
        blob_names = self._client.list_blob_names()

        blob_urls = []
        for name in blob_names:
            blob_urls.append(self._container_url + "/" + name)

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
                self._client.upload_blob(
                    name=local_file_name, data=file_data, length=sys.getsizeof(upload_file_path)
                )

        # Temporary directory is removed at this point

        # Track created blob URL
        blob_url = self._container_url + "/" + local_file_name
        self._created_blob_urls.append(blob_url)

        # Add prompt to memory
        # TODO: Need a way to record the blob url ~with~ the prompt
        # TODO: Do we want to add the blob_url to memory instead of the message?
        self.memory.add_chat_message_to_memory(
            conversation=ChatMessage(role="user", content=normalized_prompt),
            conversation_id=conversation_id,
            normalizer_id=normalizer_id,
        )

        return blob_url
    
    # TODO: This will be the way to send prompts once https://github.com/Azure/PyRIT/pull/91/ merges
    async def send_prompt_async(self, normalized_prompt: str, conversation_id: str, normalizer_id: str) -> str:

        # Create temporary file to upload to Storage Account Container URL
        local_file_name = str(uuid.uuid4()) + ".txt"

        with tempfile.TemporaryDirectory() as temp_dir:
            upload_file_path = os.path.join(temp_dir, local_file_name)

            file = open(file=upload_file_path, mode="w")
            file.write(normalized_prompt)
            file.close()

            print("\nUploading to Azure Storage as blob:\n\t" + local_file_name)

            with open(file=upload_file_path, mode="rb") as file_data:
                await self._client.upload_blob(
                    name=local_file_name, data=file_data, length=sys.getsizeof(upload_file_path)
                )

        # Track created blob URL
        blob_url = self._container_url + "/" + local_file_name
        self._created_blob_urls.append(blob_url)

        # Add prompt to memory
        # TODO: Need a way to record the blob url with the prompt
        self.memory.add_chat_message_to_memory(
            conversation=ChatMessage(role="user", content=normalized_prompt),
            conversation_id=conversation_id,
            normalizer_id=normalizer_id,
        )

        return blob_url