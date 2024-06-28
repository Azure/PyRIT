# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import json
import logging
from abc import abstractmethod
from typing import Optional

from openai import AsyncAzureOpenAI, AsyncOpenAI, AzureOpenAI, OpenAI, BadRequestError
from openai.types.chat import ChatCompletion

from pyrit.auth.azure_auth import get_token_provider_from_default_azure_credential
from pyrit.common import default_values
from pyrit.exceptions import EmptyResponseException, PyritException
from pyrit.exceptions import pyrit_target_retry, handle_bad_request_exception
from pyrit.memory import MemoryInterface
from pyrit.models import ChatMessage, PromptRequestPiece, PromptRequestResponse
from pyrit.models import construct_response_from_request
from pyrit.prompt_target import PromptChatTarget

logger = logging.getLogger(__name__)


class OpenAIChatInterface(PromptChatTarget):

    _top_p: int
    _deployment_name: str
    _temperature: float
    _frequency_penalty: float
    _presence_penalty: float
    _client: OpenAI
    _async_client: AsyncOpenAI

    @abstractmethod
    def __init__(self) -> None:
        """
        Abstract openai chat target. Must set private variables applicably
        """
        pass

    async def send_prompt_async(self, *, prompt_request: PromptRequestResponse) -> PromptRequestResponse:
        self._validate_request(prompt_request=prompt_request)
        request: PromptRequestPiece = prompt_request.request_pieces[0]

        messages = self._memory.get_chat_messages_with_conversation_id(conversation_id=request.conversation_id)
        messages.append(request.to_chat_message())

        logger.info(f"Sending the following prompt to the prompt target: {request}")

        try:
            resp_text = await self._complete_chat_async(
                messages=messages,
                top_p=self._top_p,
                temperature=self._temperature,
                frequency_penalty=self._frequency_penalty,
                presence_penalty=self._presence_penalty,
            )

            logger.info(f'Received the following response from the prompt target "{resp_text}"')
            response_entry = construct_response_from_request(request=request, response_text_pieces=[resp_text])
        except BadRequestError as bre:
            response_entry = handle_bad_request_exception(response_text=bre.message, request=request)

        return response_entry

    def _parse_chat_completion(self, response):
        """
        Parses chat message to get response

        Args:
            response (ChatMessage): The chat messages object containing the generated response message

        Returns:
            str: The generated response message
        """
        response_message = response.choices[0].message.content
        return response_message

    @pyrit_target_retry
    async def _complete_chat_async(
        self,
        messages: list[ChatMessage],
        max_tokens: int = 1024,
        temperature: float = 1.0,
        top_p: int = 1,
        frequency_penalty: float = 0.5,
        presence_penalty: float = 0.5,
    ) -> str:
        """
        Completes asynchronous chat request.

        Sends a chat message to the OpenAI chat model and retrieves the generated response.

        Args:
            messages (list[ChatMessage]): The chat message objects containing the role and content.
            max_tokens (int, optional): The maximum number of tokens to generate.
                Defaults to 1024.
            temperature (float, optional): Controls randomness in the response generation.
                Defaults to 1.0.
            top_p (int, optional): Controls diversity of the response generation.
                Defaults to 1.
            frequency_penalty (float, optional): Controls the frequency of generating the same lines of text.
                Defaults to 0.5.
            presence_penalty (float, optional): Controls the likelihood to talk about new topics.
                Defaults to 0.5.

        Returns:
            str: The generated response message.
        """

        response: ChatCompletion = await self._async_client.chat.completions.create(
            model=self._deployment_name,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            n=1,
            stream=False,
            messages=[{"role": msg.role, "content": msg.content} for msg in messages],  # type: ignore
        )
        finish_reason = response.choices[0].finish_reason
        extracted_response: str = ""
        # finish_reason="stop" means API returned complete message and
        # "length" means API returned incomplete message due to max_tokens limit.
        if finish_reason in ["stop", "length"]:
            extracted_response = self._parse_chat_completion(response)
            # Handle empty response
            if not extracted_response:
                raise EmptyResponseException(message="The chat returned an empty response.")
        else:
            raise PyritException(message=f"Unknown finish_reason {finish_reason}")
        return extracted_response

    def _validate_request(self, *, prompt_request: PromptRequestResponse) -> None:
        if len(prompt_request.request_pieces) != 1:
            raise ValueError("This target only supports a single prompt request piece.")

        if prompt_request.request_pieces[0].converted_value_data_type != "text":
            raise ValueError("This target only supports text prompt input.")


class AzureOpenAIChatTarget(OpenAIChatInterface):
    API_KEY_ENVIRONMENT_VARIABLE: str = "AZURE_OPENAI_CHAT_KEY"
    ENDPOINT_URI_ENVIRONMENT_VARIABLE: str = "AZURE_OPENAI_CHAT_ENDPOINT"
    DEPLOYMENT_ENVIRONMENT_VARIABLE: str = "AZURE_OPENAI_CHAT_DEPLOYMENT"

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
    ) -> None:
        """
        Class that initializes an Azure OpenAI chat target.

        Note that this is different from the Azure OpenAI completion target.

        Args:
            deployment_name (str, optional): The name of the deployment. Defaults to the
                AZURE_OPENAI_CHAT_DEPLOYMENT environment variable .
            endpoint (str, optional): The endpoint URL for the Azure OpenAI service.
                Defaults to the AZURE_OPENAI_CHAT_ENDPOINT environment variable.
            api_key (str, optional): The API key for accessing the Azure OpenAI service.
                Defaults to the AZURE_OPENAI_CHAT_KEY environment variable.
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
        PromptChatTarget.__init__(self, memory=memory)

        self._max_tokens = max_tokens
        self._temperature = temperature
        self._top_p = top_p
        self._frequency_penalty = frequency_penalty
        self._presence_penalty = presence_penalty

        self._deployment_name = default_values.get_required_value(
            env_var_name=self.DEPLOYMENT_ENVIRONMENT_VARIABLE, passed_value=deployment_name
        )
        endpoint = default_values.get_required_value(
            env_var_name=self.ENDPOINT_URI_ENVIRONMENT_VARIABLE, passed_value=endpoint
        )

        if use_aad_auth:
            logger.info("Authenticating with DefaultAzureCredential() for Azure Cognitive Services")
            token_provider = get_token_provider_from_default_azure_credential()

            self._client = AzureOpenAI(
                azure_ad_token_provider=token_provider,
                api_version=api_version,
                azure_endpoint=endpoint,
            )
            self._async_client = AsyncAzureOpenAI(
                azure_ad_token_provider=token_provider,
                api_version=api_version,
                azure_endpoint=endpoint,
            )
        else:
            api_key = default_values.get_required_value(
                env_var_name=self.API_KEY_ENVIRONMENT_VARIABLE, passed_value=api_key
            )

            self._client = AzureOpenAI(
                api_key=api_key,
                api_version=api_version,
                azure_endpoint=endpoint,
            )
            self._async_client = AsyncAzureOpenAI(
                api_key=api_key,
                api_version=api_version,
                azure_endpoint=endpoint,
            )


class OpenAIChatTarget(OpenAIChatInterface):
    API_KEY_ENVIRONMENT_VARIABLE: str = "OPENAI_CHAT_KEY"
    ENDPOINT_URI_ENVIRONMENT_VARIABLE: str = "OPENAI_CHAT_ENDPOINT"
    DEPLOYMENT_ENVIRONMENT_VARIABLE: str = "OPENAI_CHAT_DEPLOYMENT"

    def __init__(
        self,
        *,
        deployment_name: str = None,
        endpoint: str = None,
        api_key: str = None,
        memory: MemoryInterface = None,
        max_tokens: int = 1024,
        temperature: float = 1.0,
        top_p: int = 1,
        frequency_penalty: float = 0.5,
        presence_penalty: float = 0.5,
        headers: Optional[dict[str, str]] = None,
    ) -> None:
        """
        Class that initializes an openai chat target

        Args:
            deployment_name (str, optional): The name of the deployment. Defaults to the
                DEPLOYMENT_ENVIRONMENT_VARIABLE environment variable .
            endpoint (str, optional): The endpoint URL for the Azure OpenAI service.
                Defaults to the ENDPOINT_URI_ENVIRONMENT_VARIABLE environment variable.
            api_key (str, optional): The API key for accessing the Azure OpenAI service.
                Defaults to the API_KEY_ENVIRONMENT_VARIABLE environment variable.
            memory (MemoryInterface, optional): An instance of the MemoryInterface class
                for storing conversation history. Defaults to None.
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
        PromptChatTarget.__init__(self, memory=memory)

        self._max_tokens = max_tokens
        self._temperature = temperature
        self._top_p = top_p
        self._frequency_penalty = frequency_penalty
        self._presence_penalty = presence_penalty

        self._deployment_name = default_values.get_required_value(
            env_var_name=self.DEPLOYMENT_ENVIRONMENT_VARIABLE, passed_value=deployment_name
        )
        endpoint = default_values.get_required_value(
            env_var_name=self.ENDPOINT_URI_ENVIRONMENT_VARIABLE, passed_value=endpoint
        )
        api_key = default_values.get_required_value(
            env_var_name=self.API_KEY_ENVIRONMENT_VARIABLE, passed_value=api_key
        )
        if headers:
            self._client = OpenAI(api_key=api_key, base_url=endpoint, default_headers=json.loads(str(headers)))
        else:
            self._client = OpenAI(
                api_key=api_key,
                base_url=endpoint,
            )
        self._async_client = AsyncOpenAI(
            api_key=api_key,
            base_url=endpoint,
        )
