# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import re

from pyrit.common import default_values
from pyrit.memory import MemoryInterface
from pyrit.prompt_target.prompt_chat_target.azure_ml_chat_target import AzureMLChatTarget

from azure.core.exceptions import ClientAuthenticationError
from azure.storage.blob import ContainerClient
import logging


logger = logging.getLogger(__name__)


class RetrievalEnabledPromptTarget(AzureMLChatTarget):
    """A prompt target that can retrieve content.
    
    Not all prompt targets are able to retrieve content.
    For example, LLM endpoints in Azure do not have permission to make queries to the internet.
    This abstract class expands on the PromptTarget definition to include the ability to retrieve content.
    What kind of content is retrieved is up to the implementation.
    This could be files from storage blobs, pages from the internet, emails from a mail server, etc.
    If an endpoint is able to retrieve content by itself (by specifying the location in the prompt)
    then it should inherit from this class but implement the retrieve_content method and return None.

    Args:
        container_url (str): URL to the Azure Blob Storage Container.
        sas_token (str): Blob SAS token required to authenticate blob operations.
        memory (str): MemoryInterface to use for the class. FileMemory by default.
    """
    AZURE_STORAGE_CONTAINER_ENVIRONMENT_VARIABLE: str = "AZURE_STORAGE_ACCOUNT_CONTAINER_URL"
    SAS_TOKEN_ENVIRONMENT_VARIABLE: str = "AZURE_STORAGE_ACCOUNT_SAS_TOKEN"

    def __init__(
        self,
        *,
        container_url: str | None = None,
        sas_token: str | None = None,
        memory: MemoryInterface
    ) -> None:
        self._container_url: str = default_values.get_required_value(
            env_var_name=self.AZURE_STORAGE_CONTAINER_ENVIRONMENT_VARIABLE, passed_value=container_url
        )

        self._sas_token: str = default_values.get_required_value(
            env_var_name=self.SAS_TOKEN_ENVIRONMENT_VARIABLE, passed_value=sas_token
        )

        self._storage_client = ContainerClient.from_container_url(
            container_url=self._container_url,
            credential=self._sas_token,
        )

        # self._client_async = AsyncContainerClient.from_container_url(
        #     container_url=self._container_url,
        #     credential=self._sas_token,
        # )
        super().__init__(memory=memory)

    def send_prompt(self, *, normalized_prompt: str, conversation_id: str, normalizer_id: str) -> str:
        match = re.search(r"[-\w]+.txt", normalized_prompt)

        if not match:
            raise ValueError("Prompt does not contain a file name.")
        
        file_name = match.group()
        logger.info(
            f"Retrieving content from Azure Storage blob with file name: {file_name}"
        )
        print(file_name)
        content = self._download_blob(file_name=file_name)

        if not content:
            raise ValueError("File is empty.")

        normalized_prompt += f"\n\nBelow is the content of file {file_name}:\n\n{content}"

        print(normalized_prompt)

        return super().send_prompt(normalized_prompt=normalized_prompt, conversation_id=conversation_id, normalizer_id=normalizer_id)

    def _download_blob_exception_handling(self, exc: Exception) -> None:
        """
        Handles exceptions for uploading blob to storage container.

        Raises:
            ClientAuthenticationError: If authentication fails, either from an invalid SAS token or from
                an invalid container URL.
            Exception: If anything except ClientAuthenticationError is caught when uploading a blob.
        """

        if type(exc) is ClientAuthenticationError:
            logger.exception(
                msg="Authentication failed. Verify the container's existence in the Azure Storage Account and "
                + "the validity of the provided SAS token."
            )
            raise
        else:
            logger.exception(msg=f"An unexpected error occurred: {exc}")
            raise

    def _download_blob(self, file_name: str) -> None:
        """
        Handles uploading blob to given storage container.

        Args:
            file_name (str): File name to assign to uploaded blob.
        """

        logger.info(msg="\nDownloading from Azure Storage blob:\n\t" + file_name)

        try:
            return self._storage_client.get_blob_client(blob=file_name).download_blob().readall().decode("utf-8")
            # self._storage_client.download_blob(blob=file_name, encoding="UTF-8").readall()
        except Exception as exc:
            self._download_blob_exception_handling(exc=exc)
