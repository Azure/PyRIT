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
    PromptRequestPiece,
    PromptRequestResponse,
)
from pyrit.prompt_target import OpenAITarget, limit_requests_per_minute

logger = logging.getLogger(__name__)


class OpenAIChatTargetBase(OpenAITarget):
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
            use_aad_auth (bool, Optional): When set to True, user authentication is used
                instead of API Key. DefaultAzureCredential is taken for
                https://cognitiveservices.azure.com/.default . Please run `az login` locally
                to leverage user AuthN.
            api_version (str, Optional): The version of the Azure OpenAI API. Defaults to
                "2024-10-21".
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
            raise PyritException("temperature must be between 0 and 2 (inclusive).")
        if top_p is not None and (top_p < 0 or top_p > 1):
            raise PyritException("top_p must be between 0 and 1 (inclusive).")

        self._temperature = temperature
        self._top_p = top_p
        self._is_json_supported = is_json_supported
        self._extra_body_parameters = extra_body_parameters

    @limit_requests_per_minute
    @pyrit_target_retry
    async def send_prompt_async(self, *, prompt_request: PromptRequestResponse) -> PromptRequestResponse:
        """Asynchronously sends a prompt request and handles the response within a managed conversation context.

        Args:
            prompt_request (PromptRequestResponse): The prompt request response object.

        Returns:
            PromptRequestResponse: The updated conversation entry with the response from the prompt target.
        """

        self._validate_request(prompt_request=prompt_request)
        self.refresh_auth_headers()

        request_piece: PromptRequestPiece = prompt_request.request_pieces[0]

        is_json_response = self.is_response_format_json(request_piece)

        conversation = self._memory.get_conversation(conversation_id=request_piece.conversation_id)
        conversation.append(prompt_request)

        logger.info(f"Sending the following prompt to the prompt target: {prompt_request}")

        body = await self._construct_request_body(conversation=conversation, is_json_response=is_json_response)

        params = {}
        if self._api_version is not None:
            params["api-version"] = self._api_version

        try:
            str_response: httpx.Response = await net_utility.make_request_and_raise_if_error_async(
                endpoint_uri=self._endpoint,
                method="POST",
                headers=self._headers,
                request_body=body,
                params=params,
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
                    return handle_bad_request_exception(
                        response_text=error_response_text,
                        request=request_piece,
                        error_code=StatusError.response.status_code,
                        is_content_filter=is_content_filter,
                    )

                except json.JSONDecodeError:
                    # Not valid JSON, proceed without parsing
                    return handle_bad_request_exception(
                        response_text=error_response_text,
                        request=request_piece,
                        error_code=StatusError.response.status_code,
                    )
            elif StatusError.response.status_code == 429:
                raise RateLimitException()
            else:
                raise

        logger.info(f'Received the following response from the prompt target "{str_response.text}"')
        response: PromptRequestResponse = self._construct_prompt_response_from_openai_json(
            open_ai_str_response=str_response.text, request_piece=request_piece
        )

        return response

    async def _construct_request_body(
        self, conversation: MutableSequence[PromptRequestResponse], is_json_response: bool
    ) -> dict:
        raise NotImplementedError

    def _construct_prompt_response_from_openai_json(
        self,
        *,
        open_ai_str_response: str,
        request_piece: PromptRequestPiece,
    ) -> PromptRequestResponse:
        raise NotImplementedError

    def is_json_response_supported(self) -> bool:
        """Indicates that this target supports JSON response format."""
        return self._is_json_supported
