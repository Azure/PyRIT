# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging

from openai import AsyncAzureOpenAI
from openai.types.completion import Completion

from pyrit.auth.azure_auth import get_token_provider_from_default_azure_credential
from pyrit.common import default_values
from pyrit.memory import MemoryInterface
from pyrit.models import PromptResponse
from pyrit.models.prompt_request_response import PromptRequestResponse, construct_response_from_request
from pyrit.prompt_target.prompt_target import PromptTarget

logger = logging.getLogger(__name__)


class AzureOpenAICompletionTarget(PromptTarget):
    API_KEY_ENVIRONMENT_VARIABLE: str = "AZURE_OPENAI_COMPLETION_KEY"
    ENDPOINT_URI_ENVIRONMENT_VARIABLE: str = "AZURE_OPENAI_COMPLETION_ENDPOINT"
    DEPLOYMENT_ENVIRONMENT_VARIABLE: str = "AZURE_OPENAI_COMPLETION_DEPLOYMENT"

    def __init__(
        self,
        *,
        deployment_name: str = None,
        endpoint: str = None,
        api_key: str = None,
        use_aad_auth: bool = False,
        memory: MemoryInterface = None,
        api_version: str = "2024-02-01",
        max_tokens: int = 1024,
        temperature: float = 1.0,
        top_p: int = 1,
        frequency_penalty: float = 0.5,
        presence_penalty: float = 0.5,
    ):
        """
        Class that initializes an Azure OpenAI completion target.

        Note that this is different from the Azure OpenAI chat target.

        Args:
            deployment_name (str, optional): The name of the deployment. Defaults to the
                AZURE_OPENAI_COMPLETION_DEPLOYMENT environment variable .
            endpoint (str, optional): The endpoint URL for the Azure OpenAI service.
                Defaults to the AZURE_OPENAI_COMPLETION_ENDPOINT environment variable.
            api_key (str, optional): The API key for accessing the Azure OpenAI service.
                Defaults to the AZURE_OPENAI_COMPLETION_KEY environment variable.
            use_aad_auth (bool, optional): When set to True, user authentication is used
                instead of API Key. DefaultAzureCredential is taken for
                https://cognitiveservices.azure.com/.default. Please run `az login` locally
                to leverage user AuthN.
            memory (MemoryInterface, optional): An instance of the MemoryInterface class
                for storing conversation history. Defaults to None.
            api_version (str, optional): The version of the Azure OpenAI API. Defaults to
                "2024-02-01".
            max_tokens (int, optional): The maximum number of tokens to generate in the response.
                Defaults to 1024.
            temperature (float, optional): The temperature parameter for controlling the
                randomness of the response. Defaults to 1.0.
            top_p (int, optional): The top-p parameter for controlling the diversity of the
                response. Defaults to 1.
            frequency_penalty (float, optional): The frequency penalty parameter for penalizing
                frequently generated tokens. Defaults to 0.5.
            presence_penalty (float, optional): The presence penalty parameter for penalizing
                tokens that are already present in the conversation history. Defaults to 0.5.
        """
        super().__init__(memory=memory)

        self._max_tokens = max_tokens
        self._temperature = temperature
        self._top_p = top_p
        self._frequency_penalty = frequency_penalty
        self._presence_penalty = presence_penalty

        endpoint = default_values.get_required_value(
            env_var_name=self.ENDPOINT_URI_ENVIRONMENT_VARIABLE, passed_value=endpoint
        )
        self._model = default_values.get_required_value(
            env_var_name=self.DEPLOYMENT_ENVIRONMENT_VARIABLE, passed_value=deployment_name
        )

        if use_aad_auth:
            logger.info("Authenticating with DefaultAzureCredential() for Azure Cognitive Services")
            token_provider = get_token_provider_from_default_azure_credential()

            self._async_client = AsyncAzureOpenAI(
                azure_ad_token_provider=token_provider,
                api_version=api_version,
                azure_endpoint=endpoint,
            )
        else:
            api_key = default_values.get_required_value(
                env_var_name=self.API_KEY_ENVIRONMENT_VARIABLE, passed_value=api_key
            )

            self._async_client = AsyncAzureOpenAI(
                api_key=api_key,
                api_version=api_version,
                azure_endpoint=endpoint,
            )

    async def send_prompt_async(self, *, prompt_request: PromptRequestResponse) -> PromptRequestResponse:
        """
        Sends a normalized prompt async to the prompt target.
        """
        self._validate_request(prompt_request=prompt_request)
        request = prompt_request.request_pieces[0]

        logger.info(f"Sending the following prompt to the prompt target: {request}")

        text_response: Completion = await self._async_client.completions.create(
            model=self._model,
            prompt=request.converted_value,
            top_p=self._top_p,
            temperature=self._temperature,
            frequency_penalty=self._frequency_penalty,
            presence_penalty=self._presence_penalty,
            max_tokens=self._max_tokens,
        )
        prompt_response = PromptResponse(
            completion=text_response.choices[0].text,
            prompt=request.converted_value,
            id=text_response.id,
            completion_tokens=text_response.usage.completion_tokens,
            prompt_tokens=text_response.usage.prompt_tokens,
            total_tokens=text_response.usage.total_tokens,
            model=text_response.model,
            object=text_response.object,
        )
        response_entry = construct_response_from_request(
            request=request,
            response_text_pieces=[prompt_response.completion],
            prompt_metadata=prompt_response.to_json(),
        )

        return response_entry

    def _validate_request(self, *, prompt_request: PromptRequestResponse) -> None:
        if len(prompt_request.request_pieces) != 1:
            raise ValueError("This target only supports a single prompt request piece.")

        if prompt_request.request_pieces[0].converted_value_data_type != "text":
            raise ValueError("This target only supports text prompt input.")

        request = prompt_request.request_pieces[0]
        messages = self._memory.get_chat_messages_with_conversation_id(conversation_id=request.conversation_id)

        if len(messages) > 0:
            raise ValueError("This target only supports a single turn conversation.")
