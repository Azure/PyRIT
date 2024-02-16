# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging

from pyrit.common import default_values, net_utility
from pyrit.chat_message_normalizer import ChatMessageNormalizer, ChatMessageNop
from pyrit.interfaces import ChatSupport
from pyrit.models import ChatMessage

logger = logging.getLogger(__name__)


class AMLOnlineEndpointChat(ChatSupport):
    """
    The AMLOnlineEndpointChat interacts with AML-managed online endpoints, specifically
    for conducting red teaming activities.
    """

    API_KEY_ENVIRONMENT_VARIABLE: str = "AZURE_ML_API_KEY"
    ENDPOINT_URI_ENVIRONMENT_VARIABLE: str = "AZURE_ML_MANAGED_ENDPOINT"

    def __init__(
        self,
        *,
        endpoint_uri: str = None,
        api_key: str = None,
        chat_message_normalizer: ChatMessageNormalizer = ChatMessageNop(),
    ) -> None:
        """
        Args:
            endpoint_uri: AML online endpoint URI.
            api_key: api key for the endpoint
            chat_message_normalizer: The chat message normalizer to use. Some models expect
                different formats for system prompts and this class provides that
        """
        self.endpoint_uri: str = default_values.get_required_value(
            env_var_name=self.ENDPOINT_URI_ENVIRONMENT_VARIABLE, passed_value=endpoint_uri
        )
        self.api_key: str = default_values.get_required_value(
            env_var_name=self.API_KEY_ENVIRONMENT_VARIABLE, passed_value=api_key
        )
        self.chat_message_normalizer = chat_message_normalizer

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
        messages_dict = [message.dict() for message in squashed_messages]

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
