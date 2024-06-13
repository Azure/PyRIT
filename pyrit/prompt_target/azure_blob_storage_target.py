# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging

from azure.core.exceptions import ClientAuthenticationError
from azure.storage.blob.aio import ContainerClient as AsyncContainerClient
from azure.storage.blob import ContentSettings
from enum import Enum

from pyrit.common import default_values
from pyrit.memory import MemoryInterface
from pyrit.models import PromptRequestResponse
from pyrit.models.prompt_request_response import construct_response_from_request
from pyrit.prompt_target import PromptTarget

logger = logging.getLogger(__name__)


class SupportedContentType(Enum):
    """
    All supported content types for uploading blobs to provided storage account container.
    See all options here: https://www.iana.org/assignments/media-types/media-types.xhtml
    """

    PLAIN_TEXT = "text/plain"


class AzureBlobStorageTarget(PromptTarget):
    """
    The AzureBlobStorageTarget takes prompts, saves the prompts to a file, and stores them as a blob in a provided
    storage account container.

    Args:
        container_url (str): URL to the Azure Blob Storage Container.
        sas_token (str): Blob SAS token required to authenticate blob operations.
        blob_content_type (SupportedContentType): Expected Content Type of the blob, chosen from the
            SupportedContentType enum. Set to PLAIN_TEXT by default.
        memory (str): MemoryInterface to use for the class. FileMemory by default.
    """

    AZURE_STORAGE_CONTAINER_ENVIRONMENT_VARIABLE: str = "AZURE_STORAGE_ACCOUNT_CONTAINER_URL"
    SAS_TOKEN_ENVIRONMENT_VARIABLE: str = "AZURE_STORAGE_ACCOUNT_SAS_TOKEN"

    def __init__(
        self,
        *,
        container_url: str | None = None,
        sas_token: str | None = None,
        blob_content_type: SupportedContentType = SupportedContentType.PLAIN_TEXT,
        memory: MemoryInterface | None = None,
    ) -> None:

        self._blob_content_type: str = blob_content_type.value

        self._container_url: str = default_values.get_required_value(
            env_var_name=self.AZURE_STORAGE_CONTAINER_ENVIRONMENT_VARIABLE, passed_value=container_url
        )

        self._sas_token: str = default_values.get_required_value(
            env_var_name=self.SAS_TOKEN_ENVIRONMENT_VARIABLE, passed_value=sas_token
        )

        self._client_async = AsyncContainerClient.from_container_url(
            container_url=self._container_url,
            credential=self._sas_token,
        )

        super().__init__(memory=memory)

    async def _upload_blob_async(self, file_name: str, data: bytes, content_type: str) -> None:
        """
        (Async) Handles uploading blob to given storage container.

        Args:
            file_name (str): File name to assign to uploaded blob.
            data (bytes): Byte representation of content to upload to container.
            content_type (str): Content type to upload.
        """

        content_settings = ContentSettings(content_type=f"{content_type}")
        logger.info(msg="\nUploading to Azure Storage as blob:\n\t" + file_name)

        try:
            await self._client_async.upload_blob(
                name=file_name,
                data=data,
                content_settings=content_settings,
                overwrite=True,
            )
        except Exception as exc:
            if type(exc) is ClientAuthenticationError:
                logger.exception(
                    msg="Authentication failed. Verify the container's existence in the Azure Storage Account and "
                    + "the validity of the provided SAS token."
                )
                raise
            else:
                logger.exception(msg=f"An unexpected error occurred: {exc}")
                raise

    async def send_prompt_async(self, *, prompt_request: PromptRequestResponse) -> PromptRequestResponse:
        """
        (Async) Sends prompt to target, which creates a file and uploads it as a blob
        to the provided storage container.

        Args:
            normalized_prompt (str): A normalized prompt to be sent to the prompt target.
            conversation_id (str): The ID of the conversation.
            normalizer_id (str): ID provided by the prompt normalizer.

        Returns:
            blob_url (str): The Blob URL of the created blob within the provided storage container.
        """
        self._validate_request(prompt_request=prompt_request)
        request = prompt_request.request_pieces[0]

        file_name = f"{request.conversation_id}.txt"
        data = str.encode(request.converted_value)
        blob_url = self._container_url + "/" + file_name

        await self._upload_blob_async(file_name=file_name, data=data, content_type=self._blob_content_type)

        response = construct_response_from_request(
            request=request, response_text_pieces=[blob_url], response_type="url"
        )

        return response

    def _validate_request(self, *, prompt_request: PromptRequestResponse) -> None:
        if len(prompt_request.request_pieces) != 1:
            raise ValueError("This target only supports a single prompt request piece.")

        if prompt_request.request_pieces[0].converted_value_data_type not in ["text", "url"]:
            raise ValueError("This target only supports text and url prompt input.")

        request = prompt_request.request_pieces[0]
        messages = self._memory.get_chat_messages_with_conversation_id(conversation_id=request.conversation_id)

        if len(messages) > 0:
            raise ValueError("This target only supports a single turn conversation.")
