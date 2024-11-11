from typing import Any, Optional

from azure.storage.blob.aio import ContainerClient as AsyncContainerClient
import logging
from openai import AsyncAzureOpenAI
from semantic_kernel.kernel import Kernel
from semantic_kernel.connectors.ai.open_ai.prompt_execution_settings.azure_chat_prompt_execution_settings import (
    AzureChatPromptExecutionSettings,
)
from semantic_kernel.connectors.ai.open_ai.services.azure_chat_completion import AzureChatCompletion
from semantic_kernel.prompt_template.prompt_template_config import PromptTemplateConfig
from semantic_kernel.functions.kernel_function_decorator import kernel_function

from pyrit.auth import AzureStorageAuth
from pyrit.common import default_values
from pyrit.models import PromptRequestResponse
from pyrit.models import construct_response_from_request
from pyrit.prompt_target import PromptChatTarget

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

    def send_prompt(self, *, prompt_request: PromptRequestResponse) -> PromptRequestResponse:

        raise NotImplementedError("SemanticKernelPluginPromptTarget only supports send_prompt_async")

    def set_system_prompt(
        self,
        *,
        system_prompt: str,
        conversation_id: str,
        orchestrator_identifier: Optional[dict[str, str]] = None,
        labels: Optional[dict[str, str]] = None,
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
        self._validate_request(prompt_request=prompt_request)

        request = prompt_request.request_pieces[0]

        logger.info(f"Processing: {prompt_request}")
        prompt_template_config = PromptTemplateConfig(
            template=request.converted_value,
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

        response = construct_response_from_request(request=request, response_text_pieces=[processing_output])
        return response

    def _validate_request(self, *, prompt_request: PromptRequestResponse) -> None:
        if len(prompt_request.request_pieces) != 1:
            raise ValueError("This target only supports a single prompt request piece.")

        if prompt_request.request_pieces[0].converted_value_data_type != "text":
            raise ValueError("This target only supports text prompt input.")

        request = prompt_request.request_pieces[0]
        messages = self._memory.get_chat_messages_with_conversation_id(conversation_id=request.conversation_id)

        if len(messages) > 0:
            raise ValueError("This target only supports a single turn conversation.")


class AzureStoragePlugin:
    AZURE_STORAGE_CONTAINER_ENVIRONMENT_VARIABLE: str = "AZURE_STORAGE_ACCOUNT_CONTAINER_URL"
    SAS_TOKEN_ENVIRONMENT_VARIABLE: str = "AZURE_STORAGE_ACCOUNT_SAS_TOKEN"

    def __init__(self, *, container_url: str | None = None, sas_token: Optional[str] = None) -> None:

        self._container_url: str = default_values.get_required_value(
            env_var_name=self.AZURE_STORAGE_CONTAINER_ENVIRONMENT_VARIABLE, passed_value=container_url
        )
        self._sas_token = sas_token
        self._storage_client: AsyncContainerClient = None

    async def _create_container_client_async(self) -> None:
        """Creates an asynchronous ContainerClient for Azure Storage. If a SAS token is provided via the
        AZURE_STORAGE_ACCOUNT_SAS_TOKEN environment variable or the init sas_token parameter, it will be used
        for authentication. Otherwise, a delegation SAS token will be created using Entra ID authentication."""
        try:
            sas_token: str = default_values.get_required_value(
                env_var_name=self.SAS_TOKEN_ENVIRONMENT_VARIABLE, passed_value=self._sas_token
            )
            logger.info("Using SAS token from environment variable or passed parameter.")
        except ValueError:
            logger.info("SAS token not provided. Creating a delegation SAS token using Entra ID authentication.")
            sas_token = await AzureStorageAuth.get_sas_token(self._container_url)
        self._storage_client = AsyncContainerClient.from_container_url(
            container_url=self._container_url,
            credential=sas_token,
        )

    @kernel_function(
        description="Retrieves blobs from Azure storage asynchronously",
        name="download_async",
    )
    async def download_async(self) -> str:
        if not self._storage_client:
            await self._create_container_client_async()

        all_blobs = ""
        async with self._storage_client as client:
            async for blob in client.list_blobs():
                logger.info(f"Downloading Azure storage blob {blob.name}")
                blob_client = client.get_blob_client(blob=blob.name)
                blob_data = await blob_client.download_blob()
                blob_content = await blob_data.readall()
                all_blobs += f"\n\nBlob: {blob.name}\n"
                all_blobs += blob_content.decode("utf-8")
        logger.info(f"Azure storage download result: {all_blobs}")
        return all_blobs

    async def delete_blobs_async(self):
        """
        (Async) Deletes all blobs in the storage container.
        """
        if not self._storage_client:
            await self._create_container_client_async()
        logger.info("Deleting all blobs in the container.")
        try:
            async with self._storage_client as client:
                async for blob in client.list_blobs():
                    print("blob name is given as", blob.name)
                    await client.get_blob_client(blob=blob.name).delete_blob()
                    logger.info(f"Deleted blob: {blob.name}")
        except Exception as ex:
            logger.exception(msg=f"An error occurred while deleting blobs: {ex}")
            raise
