# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging
from typing import Any, Optional

from pyrit.chat_message_normalizer import ChatMessageNop, ChatMessageNormalizer
from pyrit.common import default_values, net_utility
from pyrit.models import (
    ChatMessage,
    PromptRequestPiece,
    PromptRequestResponse,
    construct_response_from_request,
)
from pyrit.prompt_target import PromptChatTarget, limit_requests_per_minute

logger = logging.getLogger(__name__)


class OllamaChatTarget(PromptChatTarget):

    ENDPOINT_URI_ENVIRONMENT_VARIABLE = "OLLAMA_ENDPOINT"
    MODEL_NAME_ENVIRONMENT_VARIABLE = "OLLAMA_MODEL_NAME"
    OLLAMA_DEFAULT_REMPERATURE_VALUE = 0.8  # https://github.com/ollama/ollama/blob/main/docs/modelfile.md#parameter

    def __init__(
        self,
        *,
        endpoint: str = None,
        model_name: str = None,
        temperature: float = None,
        chat_message_normalizer: ChatMessageNormalizer = ChatMessageNop(),
        max_requests_per_minute: Optional[int] = None,
        **httpx_client_kwargs: Optional[Any],
    ) -> None:
        PromptChatTarget.__init__(self, max_requests_per_minute=max_requests_per_minute)

        self.endpoint = endpoint or default_values.get_required_value(
            env_var_name=self.ENDPOINT_URI_ENVIRONMENT_VARIABLE, passed_value=endpoint
        )
        self.model_name = model_name or default_values.get_required_value(
            env_var_name=self.MODEL_NAME_ENVIRONMENT_VARIABLE, passed_value=model_name
        )
        self.temperature = (
            float(temperature) if isinstance(temperature, (float, int)) else self.OLLAMA_DEFAULT_REMPERATURE_VALUE
        )
        self.chat_message_normalizer = chat_message_normalizer
        self.httpx_client_kwargs = httpx_client_kwargs or {}

    @limit_requests_per_minute
    async def send_prompt_async(self, *, prompt_request: PromptRequestResponse) -> PromptRequestResponse:

        self._validate_request(prompt_request=prompt_request)
        request: PromptRequestPiece = prompt_request.request_pieces[0]

        messages = self._memory.get_chat_messages_with_conversation_id(conversation_id=request.conversation_id)
        messages.append(request.to_chat_message())

        logger.info(f"Sending the following prompt to the prompt target: {self} {request}")

        resp = await self._complete_chat_async(messages=messages)

        if not resp:
            raise ValueError("The chat returned an empty response.")

        logger.info(f'Received the following response from the prompt target "{resp}"')

        return construct_response_from_request(request=request, response_text_pieces=[resp])

    async def _complete_chat_async(
        self,
        messages: list[ChatMessage],
    ) -> str:
        headers = self._get_headers()
        payload = self._construct_http_body(messages)

        response = await net_utility.make_request_and_raise_if_error_async(
            endpoint_uri=self.endpoint, method="POST", request_body=payload, headers=headers, **self.httpx_client_kwargs
        )

        return response.json()["message"]["content"]

    def _construct_http_body(
        self,
        messages: list[ChatMessage],
    ) -> dict:
        squashed_messages = self.chat_message_normalizer.normalize(messages)
        messages_list = [message.model_dump(exclude_none=True) for message in squashed_messages]
        data = {
            "model": self.model_name,
            "messages": messages_list,
            "temperature": self.temperature,
            "stream": False,
        }
        return data

    def _get_headers(self) -> dict:
        headers: dict = {
            "Content-Type": "application/json",
        }

        return headers

    def _validate_request(self, *, prompt_request: PromptRequestResponse) -> None:
        if len(prompt_request.request_pieces) != 1:
            raise ValueError("This target only supports a single prompt request piece.")

        if prompt_request.request_pieces[0].converted_value_data_type != "text":
            raise ValueError("This target only supports text prompt input.")

    def is_json_response_supported(self) -> bool:
        """Indicates that this target supports JSON response format."""
        return False
