# Copyright (c) Adriano Maia <adriano@drstrange.wtf>
# Licensed under the MIT license.

import logging

from pyrit.chat_message_normalizer import ChatMessageNop, ChatMessageNormalizer
from pyrit.common import default_values, net_utility
from pyrit.memory import MemoryInterface
from pyrit.models import ChatMessage, PromptRequestPiece, PromptRequestResponse
from pyrit.models import construct_response_from_request
from pyrit.prompt_target import PromptChatTarget


logger = logging.getLogger(__name__)


class OllamaChatTarget(PromptChatTarget):

    ENDPOINT_URI_ENVIRONMENT_VARIABLE = "OLLAMA_ENDPOINT"
    MODEL_NAME_ENVIRONMENT_VARIABLE = "OLLAMA_MODEL_NAME"

    def __init__(
        self,
        *,
        endpoint_uri: str = None,
        model_name: str = None,
        chat_message_normalizer: ChatMessageNormalizer = ChatMessageNop(),
        memory: MemoryInterface = None,
    ) -> None:
        PromptChatTarget.__init__(self, memory=memory)

        self.endpoint_uri = endpoint_uri or default_values.get_required_value(
            env_var_name=self.ENDPOINT_URI_ENVIRONMENT_VARIABLE, passed_value=endpoint_uri
        )
        self.model_name = model_name or default_values.get_required_value(
            env_var_name=self.MODEL_NAME_ENVIRONMENT_VARIABLE, passed_value=model_name
        )
        self.chat_message_normalizer = chat_message_normalizer

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
            endpoint_uri=self.endpoint_uri, method="POST", request_body=payload, headers=headers
        )

        return response.json()["message"]["content"]

    def _construct_http_body(
        self,
        messages: list[ChatMessage],
    ) -> dict:
        squased_messages = self.chat_message_normalizer.normalize(messages)
        messages_dict = [message.model_dump(exclude_none=True) for message in squased_messages]
        data = {
            "model": self.model_name,
            "messages": messages_dict,
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
