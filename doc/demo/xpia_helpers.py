# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Any

from pyrit.common import default_values
from pyrit.models.prompt_request_response import PromptRequestResponse
from pyrit.prompt_target.prompt_chat_target.prompt_chat_target import PromptChatTarget

from azure.storage.blob import ContainerClient
import logging
from openai import AsyncAzureOpenAI
from semantic_kernel.kernel import Kernel
from semantic_kernel.connectors.ai.open_ai.prompt_execution_settings.azure_chat_prompt_execution_settings import (
    AzureChatPromptExecutionSettings,
)
from semantic_kernel.connectors.ai.open_ai.services.azure_chat_completion import AzureChatCompletion
from semantic_kernel.prompt_template.prompt_template_config import PromptTemplateConfig
from semantic_kernel.functions.kernel_function_decorator import kernel_function


logger = logging.getLogger(__name__)


class SemanticKernelPluginAzureOpenAIPromptTarget(PromptChatTarget):
    """A prompt target that can retrieve content using semantic kernel plugins.

    Not all prompt targets are able to retrieve content.
    For example, LLM endpoints in Azure do not have permission to make queries to the internet.
    This class expands on the PromptTarget definition to include the ability to retrieve content.
    The plugin argument controls where the content is retrieved from.
    This could be files from storage blobs, pages from the internet, emails from a mail server, etc.

    Args:
        deployment_name (str, optional): The name of the deployment. Defaults to the
            DEPLOYMENT_ENVIRONMENT_VARIABLE environment variable .
        endpoint (str, optional): The endpoint URL for the Azure OpenAI service.
            Defaults to the ENDPOINT_URI_ENVIRONMENT_VARIABLE environment variable.
        api_key (str, optional): The API key for accessing the Azure OpenAI service.
            Defaults to the API_KEY_ENVIRONMENT_VARIABLE environment variable.
        api_version (str, optional): The version of the Azure OpenAI API. Defaults to
            "2024-02-15-preview".
        plugin (Any, required): The semantic kernel plugin to retrieve the attack medium.
        plugin_name (str, required): The name of the semantic kernel plugin.
        max_tokens (int, optional): The maximum number of tokens to generate in the response.
            Defaults to 2000.
        temperature (float, optional): The temperature parameter for controlling the
            randomness of the response. Defaults to 0.7.
    """

    API_KEY_ENVIRONMENT_VARIABLE: str = "AZURE_OPENAI_CHAT_KEY"
    ENDPOINT_URI_ENVIRONMENT_VARIABLE: str = "AZURE_OPENAI_CHAT_ENDPOINT"
    DEPLOYMENT_ENVIRONMENT_VARIABLE: str = "AZURE_OPENAI_CHAT_DEPLOYMENT"

    def __init__(
        self,
        *,
        deployment_name: str = None,
        endpoint: str = None,
        api_key: str = None,
        api_version: str = "2024-02-15-preview",
        plugin: Any,
        plugin_name: str,
        max_tokens: int = 2000,
        temperature: float = 0.7,
    ) -> None:
        self._deployment_name = default_values.get_required_value(
            env_var_name=self.DEPLOYMENT_ENVIRONMENT_VARIABLE, passed_value=deployment_name
        )
        endpoint = default_values.get_required_value(
            env_var_name=self.ENDPOINT_URI_ENVIRONMENT_VARIABLE, passed_value=endpoint
        )
        api_key = default_values.get_required_value(
            env_var_name=self.API_KEY_ENVIRONMENT_VARIABLE, passed_value=api_key
        )

        self._async_client = AsyncAzureOpenAI(
            api_key=api_key,
            api_version=api_version,
            azure_endpoint=endpoint,
        )

        self._kernel = Kernel()

        service_id = "chat"

        self._kernel.add_service(
            AzureChatCompletion(
                service_id=service_id, deployment_name=self._deployment_name, async_client=self._async_client
            ),
        )

        self._plugin_name = plugin_name
        self._kernel.import_plugin_from_object(plugin, plugin_name)

        self._execution_settings = AzureChatPromptExecutionSettings(
            service_id=service_id,
            ai_model_id=self._deployment_name,
            max_tokens=max_tokens,
            temperature=temperature,
        )

        super().__init__(memory=None)

    def send_prompt(self, *, prompt_request: PromptRequestResponse) -> PromptRequestResponse:

        raise NotImplementedError("SemanticKernelPluginPromptTarget only supports send_prompt_async")

    def set_system_prompt(
        self,
        *,
        system_prompt: str,
        conversation_id: str,
        orchestrator_identifier: dict[str, str],
        labels: dict,
    ) -> None:
        raise NotImplementedError("System prompt currently not supported.")

    async def send_prompt_async(self, *, prompt_request: PromptRequestResponse) -> PromptRequestResponse:
        """
        Processes the prompt template by invoking the plugin to retrieve content.

        Args:
            prompt_request (PromptRequestResponse): The prompt request containing the template to process.

        Returns:
            PromptRequestResponse: The processed prompt response.

        """
        self._memory.add_request_pieces_to_memory(request_pieces=prompt_request.request_pieces)

        request = prompt_request.request_pieces[0]

        logger.info(f"Processing: {prompt_request}")
        prompt_template_config = PromptTemplateConfig(
            template=request.converted_prompt_text,
            name=self._plugin_name,
            template_format="semantic-kernel",
            execution_settings=self._execution_settings,
        )
        processing_function = self._kernel.create_function_from_prompt(
            function_name="processingFunc", plugin_name=self._plugin_name, prompt_template_config=prompt_template_config
        )
        processing_output = await self._kernel.invoke(processing_function)
        processing_output = str(processing_output)
        logger.info(f'Received the following response from the prompt target "{processing_output}"')

        response = self._memory.add_response_entries_to_memory(
            request=request, response_text_pieces=[processing_output]
        )
        return response


class AzureStoragePlugin:
    AZURE_STORAGE_CONTAINER_ENVIRONMENT_VARIABLE: str = "AZURE_STORAGE_ACCOUNT_CONTAINER_URL"
    SAS_TOKEN_ENVIRONMENT_VARIABLE: str = "AZURE_STORAGE_ACCOUNT_SAS_TOKEN"

    def __init__(
        self,
        *,
        container_url: str | None = None,
        sas_token: str | None = None,
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

    @kernel_function(
        description="Retrieves blob from Azure storage",
        name="download",
    )
    def download(self) -> str:
        all_blobs = ""
        for blob in self._storage_client.list_blobs():
            logger.info(f"Downloading Azure storage blob {blob.name}")
            all_blobs += f"\n\nBlob: {blob.name}\n"
            all_blobs += self._storage_client.get_blob_client(blob=blob.name).download_blob().readall().decode("utf-8")
        logger.info(f"Azure storage download result: {all_blobs}")
        return all_blobs
