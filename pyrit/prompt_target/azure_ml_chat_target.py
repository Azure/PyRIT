# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import logging

from pyrit.chat_message_normalizer import ChatMessageNormalizer, ChatMessageNop
from pyrit.common import default_values, net_utility
from pyrit.interfaces import ChatSupport
from pyrit.memory import MemoryInterface
from pyrit.models import ChatMessage
from pyrit.prompt_target import PromptChatTarget

logger = logging.getLogger(__name__)


class AzureMLChatTarget(PromptChatTarget, ChatSupport):

    API_KEY_ENVIRONMENT_VARIABLE: str = "AZURE_ML_KEY"
    ENDPOINT_URI_ENVIRONMENT_VARIABLE: str = "AZURE_ML_MANAGED_ENDPOINT"

    def __init__(
        self,
        *,
        endpoint_uri: str = None,
        api_key: str = None,
        chat_message_normalizer: ChatMessageNormalizer = ChatMessageNop(),
        memory: MemoryInterface = None,
        max_tokens: int = 400,
        temperature: float = 1.0,
        top_p: int = 1,
        repetition_penalty: float = 1.2,
    ) -> None:
        super(self, memory=memory)

        self.endpoint_uri: str = default_values.get_required_value(
            env_var_name=self.ENDPOINT_URI_ENVIRONMENT_VARIABLE, passed_value=endpoint_uri
        )
        self.api_key: str = default_values.get_required_value(
            env_var_name=self.API_KEY_ENVIRONMENT_VARIABLE, passed_value=api_key
        )
        self.chat_message_normalizer = chat_message_normalizer

        self._max_tokens = max_tokens
        self._temperature = temperature
        self._top_p = top_p
        self._repetition_penalty = repetition_penalty

    def set_system_prompt(self, *, prompt: str, conversation_id: str, normalizer_id: str) -> None:
        messages = self._memory.get_memories_with_conversation_id(conversation_id=conversation_id)

        if messages:
            raise RuntimeError("Conversation already exists, system prompt needs to be set at the beginning")

        self._memory.add_chat_message_to_memory(
            conversation=ChatMessage(role="system", content=prompt),
            conversation_id=conversation_id,
            normalizer_id=normalizer_id,
        )

    def send_prompt(self, *, normalized_prompt: str, conversation_id: str, normalizer_id: str) -> str:
        messages = self._prepare_message(normalized_prompt, conversation_id, normalizer_id)

        resp = self.complete_chat(
            messages=messages,
            temperature=self._temperature,
            top_p=self._top_p,
            repetition_penalty=self._repetition_penalty
        )

        self._memory.add_chat_message_to_memory(
            ChatMessage(role="assistant", content=resp),
            conversation_id=conversation_id,
            normalizer_id=normalizer_id,
        )

        return resp

    async def send_prompt_async(self, *, normalized_prompt: str, conversation_id: str, normalizer_id: str) -> str:
        messages = self._prepare_message(normalized_prompt, conversation_id, normalizer_id)

        resp = await self.complete_chat_async(
                    messages=messages,
                    temperature=self._temperature,
                    top_p=self._top_p,
                    repetition_penalty=self._repetition_penalty
        )

        self._memory.add_chat_message_to_memory(
            ChatMessage(role="assistant", content=resp),
            conversation_id=conversation_id,
            normalizer_id=normalizer_id,
        )

        return resp

    def complete_chat(
        self,
        messages: list[ChatMessage],
        max_tokens: int = 400,
        temperature: float = 1.0,
        top_p: int = 1,
        repetition_penalty: float = 1.2,
    ) -> str:
        """
        Completes a chat interaction by generating a response to the given input prompt.
        This is a synchronous wrapper for the asynchronous _generate_and_extract_response method.

        Args:
            messages (list[ChatMessage]): The chat messages objects containing the role and content.
            max_tokens (int, optional): The maximum number of tokens to generate. Defaults to 400.
            temperature (float, optional): Controls randomness in the response generation. Defaults
                                           to 1.0. 1 is more random, 0 is less.
            top_p (int, optional): Controls diversity of the response generation. Defaults to 1. 1
                                   is more random, 0 is less.
            repetition_penalty (float, optional): Controls repetition in the response generation.
                                                  Defaults to 1.2.

        Raises:
            Exception: For any errors during the process.

        Returns:
            str: The generated response message.
        """

        headers = self._get_headers()
        payload = self._construct_http_body(messages, max_tokens, temperature, top_p, repetition_penalty)

        response = net_utility.make_request_and_raise_if_error(
            endpoint_uri=self.endpoint_uri, method="POST", request_body=payload, headers=headers
        )
        return response.json()["output"]

    async def complete_chat_async(
        self,
        messages: list[ChatMessage],
        max_tokens: int = 400,
        temperature: float = 1.0,
        top_p: int = 1,
        repetition_penalty: float = 1.2,
    ) -> str:
        """
        Completes a chat interaction by generating a response to the given input prompt.
        This is a synchronous wrapper for the asynchronous _generate_and_extract_response method.

        Args:
            messages (list[ChatMessage]): The chat messages objects containing the role and content.
            max_tokens (int, optional): The maximum number of tokens to generate. Defaults to 400.
            temperature (float, optional): Controls randomness in the response generation. Defaults
                                           to 1.0. 1 is more random, 0 is less.
            top_p (int, optional): Controls diversity of the response generation. Defaults to 1. 1
                                   is more random, 0 is less.
            repetition_penalty (float, optional): Controls repetition in the response generation.
                                                  Defaults to 1.2.

        Raises:
            Exception: For any errors during the process.

        Returns:
            str: The generated response message.
        """

        headers = self._get_headers()
        payload = self._construct_http_body(messages, max_tokens, temperature, top_p, repetition_penalty)

        response = await net_utility.make_request_and_raise_if_error_async(
            endpoint_uri=self.endpoint_uri, method="POST", request_body=payload, headers=headers
        )
        return response.json()["output"]


    def _prepare_message(self, normalized_prompt: str, conversation_id: str, normalizer_id: str):
        messages = self._memory.get_chat_messages_with_conversation_id(conversation_id=conversation_id)
        msg = ChatMessage(role="user", content=normalized_prompt)
        messages.append(msg)
        self._memory.add_chat_message_to_memory(msg, conversation_id, normalizer_id)
        return messages

    def _construct_http_body(
        self,
        messages: list[ChatMessage],
        max_tokens: int,
        temperature: float,
        top_p: int,
        repetition_penalty: float,
    ) -> dict:
        """Constructs the HTTP request body for the AML online endpoint."""

        squashed_messages = self.chat_message_normalizer.normalize(messages)
        messages_dict = [message.model_dump() for message in squashed_messages]
        data = {
            "input_data": {
                "input_string": messages_dict,
                "parameters": {
                    "max_new_tokens": max_tokens,
                    "temperature": temperature,
                    "top_p": top_p,
                    "top_k": 50,
                    "stop": ["</s>"],
                    "stop_sequences": ["</s>"],
                    "return_full_text": False,
                    "repetition_penalty": repetition_penalty,
                },
            }
        }
        return data

    def _get_headers(self) -> dict:
        """Headers for accessing inference endpoint deployed in AML.
        Returns:
            headers(dict): contains bearer token as AML key and content-type: JSON
        """

        headers: dict = {
            "Content-Type": "application/json",
            "Authorization": ("Bearer " + self.api_key),
        }

        return headers
