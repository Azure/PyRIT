# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import json
import logging
from typing import Any, MutableSequence, Optional

import httpx

from pyrit.common import net_utility
from pyrit.exceptions import (
    PyritException,
    handle_bad_request_exception,
    pyrit_target_retry,
)
from pyrit.exceptions.exception_classes import RateLimitException
from pyrit.models import (
    JsonResponseConfig,
    Message,
    MessagePiece,
)
from pyrit.prompt_target import (
    OpenAITarget,
    PromptChatTarget,
    limit_requests_per_minute,
)

logger = logging.getLogger(__name__)


class OpenAIChatTargetBase(OpenAITarget, PromptChatTarget):
    """
    This is the base class for multimodal (image and text) input and text output generation.

    It provides the foundation for the chat completions and response API classes.
    """

    def __init__(
        self,
        *,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        is_json_supported: bool = True,
        extra_body_parameters: Optional[dict[str, Any]] = None,
        **kwargs,
    ):
        """
        Args:
            model_name (str, Optional): The name of the model.
            endpoint (str, Optional): The target URL for the OpenAI service.
            api_key (str, Optional): The API key for accessing the Azure OpenAI service.
                Defaults to the OPENAI_CHAT_KEY environment variable.
            headers (str, Optional): Headers of the endpoint (JSON).
            use_entra_auth (bool, Optional): When set to True, user authentication is used
                instead of API Key. DefaultAzureCredential is taken for
                https://cognitiveservices.azure.com/.default . Please run `az login` locally
                to leverage user AuthN.
            max_requests_per_minute (int, Optional): Number of requests the target can handle per
                minute before hitting a rate limit. The number of requests sent to the target
                will be capped at the value provided.
            temperature (float, Optional): The temperature parameter for controlling the
                randomness of the response.
            top_p (float, Optional): The top-p parameter for controlling the diversity of the
                response.
            is_json_supported (bool, Optional): If True, the target will support formatting responses as JSON by
                setting the response_format header. Official OpenAI models all support this, but if you are using
                this target with different models, is_json_supported should be set correctly to avoid issues when
                using adversarial infrastructure (e.g. Crescendo scorers will set this flag).
            extra_body_parameters (dict, Optional): Additional parameters to be included in the request body.
            httpx_client_kwargs (dict, Optional): Additional kwargs to be passed to the
                httpx.AsyncClient() constructor.
                For example, to specify a 3 minute timeout: httpx_client_kwargs={"timeout": 180}

        Raises:
            PyritException: If the temperature or top_p values are out of bounds.
            ValueError: If the temperature is not between 0 and 2 (inclusive).
            ValueError: If the top_p is not between 0 and 1 (inclusive).
            RateLimitException: If the target is rate-limited.
            httpx.HTTPStatusError: If the request fails with a 400 Bad Request or 429 Too Many Requests error.
            json.JSONDecodeError: If the response from the target is not valid JSON.
            Exception: If the request fails for any other reason.
        """
        super().__init__(**kwargs)

        if temperature is not None and (temperature < 0 or temperature > 2):
            raise PyritException(message="temperature must be between 0 and 2 (inclusive).")
        if top_p is not None and (top_p < 0 or top_p > 1):
            raise PyritException(message="top_p must be between 0 and 1 (inclusive).")

        self._temperature = temperature
        self._top_p = top_p
        self._is_json_supported = is_json_supported
        self._extra_body_parameters = extra_body_parameters

    @limit_requests_per_minute
    @pyrit_target_retry
    async def send_prompt_async(self, *, message: Message) -> Message:
        """Asynchronously sends a message and handles the response within a managed conversation context.

        Args:
            message (Message): The message object.

        Returns:
            Message: The updated conversation entry with the response from the prompt target.
        """

        self._validate_request(message=message)
        self.refresh_auth_headers()

        message_piece: MessagePiece = message.message_pieces[0]

        json_response_config = self.get_json_response_config(message_piece=message_piece)

        conversation = self._memory.get_conversation(conversation_id=message_piece.conversation_id)
        conversation.append(message)

        logger.info(f"Sending the following prompt to the prompt target: {message}")

        body = await self._construct_request_body(conversation=conversation, json_config=json_response_config)

        try:
            str_response: httpx.Response = await net_utility.make_request_and_raise_if_error_async(
                endpoint_uri=self._endpoint,
                method="POST",
                headers=self._headers,
                request_body=body,
                **self._httpx_client_kwargs,
            )
        except httpx.HTTPStatusError as StatusError:
            if StatusError.response.status_code == 400:
                # Handle Bad Request
                error_response_text = StatusError.response.text
                # Content filter errors are handled differently from other 400 errors.
                # 400 Bad Request with content_filter error code indicates that the input was filtered
                # https://learn.microsoft.com/en-us/azure/ai-services/openai/concepts/content-filter
                try:
                    json_error = json.loads(error_response_text)
                    is_content_filter = json_error.get("error", {}).get("code") == "content_filter"
                except json.JSONDecodeError:
                    # Not valid JSON, set content filter to False
                    is_content_filter = False

                return handle_bad_request_exception(
                    response_text=error_response_text,
                    request=message_piece,
                    error_code=StatusError.response.status_code,
                    is_content_filter=is_content_filter,
                )
            elif StatusError.response.status_code == 429:
                raise RateLimitException()
            else:
                raise

        logger.info(f'Received the following response from the prompt target "{str_response.text}"')
        response: Message = self._construct_message_from_openai_json(
            open_ai_str_response=str_response.text, message_piece=message_piece
        )

        return response

    async def _construct_request_body(
        self, *, conversation: MutableSequence[Message], json_config: JsonResponseConfig
    ) -> dict:
        raise NotImplementedError

    def _construct_message_from_openai_json(
        self,
        *,
        open_ai_str_response: str,
        message_piece: MessagePiece,
    ) -> Message:
        raise NotImplementedError

    def is_json_response_supported(self) -> bool:
        """Indicates that this target supports JSON response format."""
        return self._is_json_supported
