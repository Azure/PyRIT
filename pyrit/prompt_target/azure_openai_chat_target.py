# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from openai import AsyncAzureOpenAI, AzureOpenAI
from openai.types.chat import ChatCompletion

from pyrit.common import default_values
from pyrit.interfaces import ChatSupport
from pyrit.memory import MemoryInterface
from pyrit.models import ChatMessage
from pyrit.prompt_target import PromptChatTarget


class AzureOpenAIChatTarget(ChatSupport, PromptChatTarget):
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
        temperature: float = 1.0,
    ) -> None:

        PromptChatTarget.__init__(self, memory=memory)

        self._temperature = temperature

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

    def set_system_prompt(self, *, prompt: str, conversation_id: str, normalizer_id: str) -> None:
        messages = self.memory.get_memories_with_conversation_id(conversation_id=conversation_id)

        if messages:
            raise RuntimeError("Conversation already exists, system prompt needs to be set at the beginning")

        self.memory.add_chat_message_to_memory(
            conversation=ChatMessage(role="system", content=prompt),
            conversation_id=conversation_id,
            normalizer_id=normalizer_id,
        )

    def send_prompt(self, *, normalized_prompt: str, conversation_id: str, normalizer_id: str) -> str:
        messages = self._prepare_message(normalized_prompt, conversation_id, normalizer_id)

        resp = self.complete_chat(messages=messages, temperature=self._temperature)

        self.memory.add_chat_message_to_memory(
            ChatMessage(role="assistant", content=resp),
            conversation_id=conversation_id,
            normalizer_id=normalizer_id,
        )

        return resp

    async def send_prompt_async(self, *, normalized_prompt: str, conversation_id: str, normalizer_id: str) -> str:
        messages = self._prepare_message(normalized_prompt, conversation_id, normalizer_id)

        resp = await super().complete_chat_async(messages=messages, temperature=self._temperature)

        self.memory.add_chat_message_to_memory(
            ChatMessage(role="assistant", content=resp),
            conversation_id=conversation_id,
            normalizer_id=normalizer_id,
        )

        return resp

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

    def _prepare_message(self, normalized_prompt: str, conversation_id: str, normalizer_id: str):
        messages = self.memory.get_chat_messages_with_conversation_id(conversation_id=conversation_id)
        msg = ChatMessage(role="user", content=normalized_prompt)
        messages.append(msg)
        self.memory.add_chat_message_to_memory(msg, conversation_id, normalizer_id)
        return messages
