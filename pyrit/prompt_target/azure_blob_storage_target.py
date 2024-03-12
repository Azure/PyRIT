# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from azure.storage.blob.aio import ContainerClient as AsyncContainerClient
from azure.storage.blob import ContainerClient
import sys
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
        default_values.load_default_env()

        if container_url is None:
            self._container_url = default_values.get_required_value(
                env_var_name=self.AZURE_STORAGE_CONTAINER_ENVIRONMENT_VARIABLE, passed_value=container_url
            )
        else:
            self._container_url = container_url

        if sas_token is None:
            sas_token = default_values.get_required_value(
                env_var_name=self.SAS_TOKEN_ENVIRONMENT_VARIABLE, passed_value=sas_token
            )

        self._client = ContainerClient.from_container_url(container_url=self._container_url, credential=sas_token)
        self._client_async = AsyncContainerClient.from_container_url(
            container_url=self._container_url, credential=sas_token
        )

        self._created_blob_urls: list[str] = []

        super().__init__(memory)

    # QUESTION: Is there benefit to tracking which blob urls are created with each class object?
    # If not, can remove this function and self._created_blob_urls list
    def list_created_blob_urls(self) -> list[str]:
        return self._created_blob_urls

    # QUESTION: Is this useful to orchestrators? Imagining that contents would be listed and then consumed as input.
    def list_all_blob_urls(self) -> list[str]:
        blob_names = self._client.list_blob_names()

        blob_urls = []
        for name in blob_names:
            blob_urls.append(self._container_url + "/" + name)

        return blob_urls

    def _prep_prompt_to_upload(self, normalized_prompt: str) -> tuple[str, bytes]:
        file_name = str(uuid.uuid4()) + ".txt"
        data = str.encode(normalized_prompt)
        print("\nUploading to Azure Storage as blob:\n\t" + file_name)

        return file_name, data

    def _post_upload_processing(self, blob_url: str, normalized_prompt: str, conversation_id: str, normalizer_id: str):
        self._created_blob_urls.append(blob_url)

        # QUESTION: Is this the right way / best way to associate blob_url with the prompt converted into a file?
        self.memory.add_chat_message_to_memory(
            conversation=ChatMessage(role="user", content=normalized_prompt),
            conversation_id=conversation_id,
            normalizer_id=normalizer_id,
            labels=[blob_url],
        )

    def send_prompt(self, normalized_prompt: str, conversation_id: str, normalizer_id: str) -> str:
        file_name, data = self._prep_prompt_to_upload(normalized_prompt=normalized_prompt)

        self._client.upload_blob(name=file_name, data=data, length=sys.getsizeof(data))
        blob_url = self._container_url + "/" + file_name

        self._post_upload_processing(
            blob_url=blob_url,
            normalized_prompt=normalized_prompt,
            conversation_id=conversation_id,
            normalizer_id=normalizer_id,
        )

        return blob_url

    async def send_prompt_async(self, normalized_prompt: str, conversation_id: str, normalizer_id: str) -> str:
        file_name, data = self._prep_prompt_to_upload(normalized_prompt=normalized_prompt)

        await self._client_async.upload_blob(name=file_name, data=data, length=sys.getsizeof(data))
        blob_url = self._container_url + "/" + file_name

        self._post_upload_processing(
            blob_url=blob_url,
            normalized_prompt=normalized_prompt,
            conversation_id=conversation_id,
            normalizer_id=normalizer_id,
        )

        return blob_url
