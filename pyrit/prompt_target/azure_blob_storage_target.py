# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from azure.storage.blob.aio import ContainerClient as AsyncContainerClient
from azure.storage.blob import ContainerClient, ContentSettings
import sys

from pyrit.common import default_values
from pyrit.memory import MemoryInterface
from pyrit.models import ChatMessage
from pyrit.prompt_target import PromptTarget


class AzureBlobStorageTarget(PromptTarget):
    """
    The AzureBlobStorageTarget takes prompts, saves the prompts to a file, and stores them as a blob in a provided
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
        container_url: str | None = None,
        sas_token: str | None = None,
        memory: MemoryInterface | None = None,
    ) -> None:
        default_values.load_default_env()

        self._container_url: str = default_values.get_required_value(
            env_var_name=self.AZURE_STORAGE_CONTAINER_ENVIRONMENT_VARIABLE, passed_value=container_url
        )

        self._sas_token: str = default_values.get_required_value(
            env_var_name=self.SAS_TOKEN_ENVIRONMENT_VARIABLE, passed_value=sas_token
        )

        self._client = ContainerClient.from_container_url(
            container_url=self._container_url,
            credential=self._sas_token,
        )

        self._client_async = AsyncContainerClient.from_container_url(
            container_url=self._container_url,
            credential=self._sas_token,
        )

        super().__init__(memory=memory)

    """
    Handles uploading blob to given storage container.

    Args:
        file_name: File name to assign to uploaded blob.
        data: Byte representation of content to upload to container.
        content_type: Content type to upload.
    """

    def _upload_blob(self, file_name: str, data: bytes, content_type: str) -> None:
        content_settings = ContentSettings(content_type=f"{content_type}")
        print("\nUploading to Azure Storage as blob:\n\t" + file_name)

        self._client.upload_blob(
            name=file_name, data=data, length=sys.getsizeof(data), content_settings=content_settings
        )

    def send_prompt(
        self,
        *,
        normalized_prompt: str,
        conversation_id: str,
        normalizer_id: str,
        content_type: str = "text/plain",
    ) -> str:
        file_name = f"{conversation_id}.txt"
        data = str.encode(normalized_prompt)
        blob_url = self._container_url + "/" + file_name

        self._upload_blob(file_name=file_name, data=data, content_type=content_type)

        self._memory.add_chat_message_to_memory(
            conversation=ChatMessage(role="user", content=normalized_prompt),
            conversation_id=conversation_id,
            normalizer_id=normalizer_id,
        )

        return blob_url

    """
    (Async) Handles uploading blob to given storage container.

    Args:
        file_name: File name to assign to uploaded blob.
        data: Byte representation of content to upload to container.
        content_type: Content type to upload.
    """

    async def _upload_blob_async(self, file_name: str, data: bytes, content_type: str) -> None:
        content_settings = ContentSettings(content_type=f"{content_type}")
        print("\nUploading to Azure Storage as blob:\n\t" + file_name)

        await self._client_async.upload_blob(
            name=file_name, data=data, length=sys.getsizeof(data), content_settings=content_settings
        )

    async def send_prompt_async(
        self,
        *,
        normalized_prompt: str,
        conversation_id: str,
        normalizer_id: str,
        content_type: str = "text/plain",
    ) -> str:
        file_name = f"{conversation_id}.txt"
        data = str.encode(normalized_prompt)
        blob_url = self._container_url + "/" + file_name

        await self._upload_blob_async(file_name=file_name, data=data, content_type=content_type)

        self._memory.add_chat_message_to_memory(
            conversation=ChatMessage(role="user", content=normalized_prompt),
            conversation_id=conversation_id,
            normalizer_id=normalizer_id,
        )

        return blob_url
