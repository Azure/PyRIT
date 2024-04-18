# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from abc import abstractmethod
import logging

from openai import AsyncAzureOpenAI, AsyncOpenAI, AzureOpenAI, OpenAI
from openai.types.chat import ChatCompletion

from pyrit.common import default_values
from pyrit.memory import MemoryInterface
from pyrit.models import ChatMessage
from pyrit.models import PromptRequestResponse, PromptRequestPiece
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

    def send_prompt(self, *, prompt_request: PromptRequestResponse) -> PromptRequestResponse:

        request: PromptRequestPiece = prompt_request.request_pieces[0]

        messages = self._memory.get_chat_messages_with_conversation_id(conversation_id=request.conversation_id)
        messages.append(request.to_chat_message())

        request.sequence = len(messages)
        logger.info(f"Sending the following prompt to the prompt target: {request}")

        self._memory.add_request_pieces_to_memory(request_pieces=[request])

        resp_text = self._complete_chat(
            messages=messages,
            top_p=self._top_p,
            temperature=self._temperature,
            frequency_penalty=self._frequency_penalty,
            presence_penalty=self._presence_penalty,
        )

        if not resp_text:
            raise ValueError("The chat returned an empty response.")

        logger.info(f'Received the following response from the prompt target "{resp_text}"')

        response_entry = self._memory.add_response_entries_to_memory(request=request, response_text_pieces=[resp_text])

        return response_entry

    async def send_prompt_async(self, *, prompt_request: PromptRequestResponse) -> PromptRequestResponse:

        request: PromptRequestPiece = prompt_request.request_pieces[0]

        messages = self._memory.get_chat_messages_with_conversation_id(conversation_id=request.conversation_id)
        messages.append(request.to_chat_message())

        request.sequence = len(messages)
        logger.info(f"Sending the following prompt to the prompt target: {request}")

        self._memory.add_request_pieces_to_memory(request_pieces=[request])

        resp_text = await self._complete_chat_async(
            messages=messages,
            top_p=self._top_p,
            temperature=self._temperature,
            frequency_penalty=self._frequency_penalty,
            presence_penalty=self._presence_penalty,
        )

        if not resp_text:
            raise ValueError("The chat returned an empty response.")

        logger.info(f'Received the following response from the prompt target "{resp_text}"')

        response_entry = self._memory.add_response_entries_to_memory(request=request, response_text_pieces=[resp_text])

        return response_entry

    def parse_chat_completion(self, response):
        """
        Parses chat message to get response

        Args:
            response (ChatMessage): The chat messages object containing the generated response message

        Returns:
            str: The generated response message
        """
        try:
            response_message = response.choices[0].message.content
        except KeyError as ex:
            if response.choices[0].finish_reason == "content_filter":
                raise RuntimeError(f"Azure blocked the response due to content filter. Response: {response}") from ex
            else:
                raise RuntimeError(f"Error in Azure Chat. Response: {response}") from ex
        return response_message

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
        return self.parse_chat_completion(response)

    def _complete_chat(
        self,
        messages: list[ChatMessage],
        max_tokens: int = 1024,
        temperature: float = 1.0,
        top_p: int = 1,
        frequency_penalty: float = 0.5,
        presence_penalty: float = 0.5,
    ) -> str:
        """
        Parses chat message to get response

        Args:
            messages (list[ChatMessage]): The chat message objects containing the role and content.
            max_tokens (int, optional): The maximum number of tokens to generate. Defaults to 1024.
            temperature (float, optional): Controls randomness in the response generation. Defaults to 1.0.
            top_p (int, optional): Controls diversity of the response generation. Defaults to 1.
            frequency_penalty (float, optional): Controls frequency of generating same lines of text. Defaults to 0.5.
            presence_penalty (float, optional): Controls likelihood to talk about new topics. Defaults to 0.5.

        Returns:
            str: The generated response message
        """
        response: ChatCompletion = self._client.chat.completions.create(
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
        return self.parse_chat_completion(response)


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
        memory: MemoryInterface = None,
        api_version: str = "2023-08-01-preview",
        max_tokens: int = 1024,
        temperature: float = 1.0,
        top_p: int = 1,
        frequency_penalty: float = 0.5,
        presence_penalty: float = 0.5,
    ) -> None:
        """
        Class that initializes an Azure Open AI chat target

        Args:
            deployment_name (str, optional): The name of the deployment. Defaults to the
                DEPLOYMENT_ENVIRONMENT_VARIABLE environment variable .
            endpoint (str, optional): The endpoint URL for the Azure OpenAI service.
                Defaults to the ENDPOINT_URI_ENVIRONMENT_VARIABLE environment variable.
            api_key (str, optional): The API key for accessing the Azure OpenAI service.
                Defaults to the API_KEY_ENVIRONMENT_VARIABLE environment variable.
            memory (MemoryInterface, optional): An instance of the MemoryInterface class
                for storing conversation history. Defaults to None.
            api_version (str, optional): The version of the Azure OpenAI API. Defaults to
                "2023-08-01-preview".
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
        self._client = OpenAI(
            api_key=api_key,
            base_url=endpoint,
        )
        self._async_client = AsyncOpenAI(
            api_key=api_key,
            base_url=endpoint,
        )
