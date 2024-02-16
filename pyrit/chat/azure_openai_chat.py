# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
from openai import AsyncAzureOpenAI, AzureOpenAI
from openai.types.chat import ChatCompletion
from pyrit.common import default_values

from pyrit.interfaces import ChatSupport
from pyrit.models import ChatMessage


class AzureOpenAIChat(ChatSupport):
    API_KEY_ENVIRONMENT_VARIABLE: str = "AZURE_OPENAI_API_KEY"
    ENDPOINT_URI_ENVIRONMENT_VARIABLE: str = "AZURE_OPENAI_ENDPOINT"

    def __init__(
        self,
        *,
        deployment_name: str,
        endpoint: str = None,
        api_key: str = None,
        api_version: str = "2023-08-01-preview",
    ) -> None:
        self._deployment_name = deployment_name

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
        self._asynch_client = AsyncAzureOpenAI(
            api_key=api_key,
            api_version=api_version,
            azure_endpoint=endpoint,
        )

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

    async def complete_chat_async(
        self,
        messages: list[ChatMessage],
        max_tokens: int = 1024,
        temperature: float = 1.0,
        top_p: int = 1,
        frequency_penalty: float = 0.5,
        presence_penalty: float = 0.5,
    ) -> str:
        """
        Completes asynchronous chat request
        Parses chat message to get response
        Args:
            message (list[ChatMessage]): The chat message objects containing the role and content.
            max_tokens (int, optional): The maximum number of tokens to generate. Defaults to 1024.
            temperature (float, optional): Controls randomness in the response generation. Defaults to 1.0.
            top_p (int, optional): Controls diversity of the response generation. Defaults to 1.
            frequency_penalty (float, optional): Controls frequency of generating same lines of text. Defaults to 0.5.
            presence_penalty (float, optional):  Controls likelihood to talk about new topics. Defaults to 0.5.
        Returns:
            str: The generated response message
        """
        response: ChatCompletion = await self._asynch_client.chat.completions.create(
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

    def complete_chat(
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
            message (list[ChatMessage]): The chat message objects containing the role and content.
            max_tokens (int, optional): The maximum number of tokens to generate. Defaults to 1024.
            temperature (float, optional): Controls randomness in the response generation. Defaults to 1.0.
            top_p (int, optional): Controls diversity of the response generation. Defaults to 1.
            frequency_penalty (float, optional): Controls frequency of generating same lines of text. Defaults to 0.5.
            presence_penalty (float, optional):  Controls likelihood to talk about new topics. Defaults to 0.5.
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
