# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import json
import logging
from typing import Optional

import httpx

from pyrit.common import net_utility
from pyrit.exceptions.exception_classes import (
    EmptyResponseException,
    RateLimitException,
    handle_bad_request_exception,
    pyrit_target_retry,
)
from pyrit.models import PromptRequestResponse, construct_response_from_request
from pyrit.models.prompt_request_piece import PromptRequestPiece
from pyrit.prompt_target import OpenAITarget, limit_requests_per_minute

logger = logging.getLogger(__name__)


class OpenAICompletionTarget(OpenAITarget):

    def __init__(
        self,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        presence_penalty: Optional[float] = None,
        frequency_penalty: Optional[float] = None,
        n: Optional[int] = None,
        *args,
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
                "2024-06-01".
            max_requests_per_minute (int, Optional): Number of requests the target can handle per
                minute before hitting a rate limit. The number of requests sent to the target
                will be capped at the value provided.
            httpx_client_kwargs (dict, Optional): Additional kwargs to be passed to the
                httpx.AsyncClient() constructor.
            max_tokens (int, Optional): The maximum number of tokens that can be generated in the
              completion. The token count of your prompt plus `max_tokens` cannot exceed the model's
              context length.
            temperature (float, Optional): What sampling temperature to use, between 0 and 2.
                Values like 0.8 will make the output more random, while lower values like 0.2 will
                make it more focused and deterministic.
            top_p (float, Optional): An alternative to sampling with temperature, called nucleus
                sampling, where the model considers the results of the tokens with top_p probability mass.
            presence_penalty (float, Optional): Number between -2.0 and 2.0. Positive values penalize new
                tokens based on whether they appear in the text so far, increasing the model's likelihood to
                talk about new topics.
            n (int, Optional): How many completions to generate for each prompt.
            httpx_client_kwargs (dict, Optional): Additional kwargs to be passed to the
                httpx.AsyncClient() constructor.
                For example, to specify a 3 minutes timeout: httpx_client_kwargs={"timeout": 180}
        """

        super().__init__(*args, **kwargs)

        self._max_tokens = max_tokens
        self._temperature = temperature
        self._top_p = top_p
        self._frequency_penalty = frequency_penalty
        self._presence_penalty = presence_penalty
        self._n = n

    def _set_openai_env_configuration_vars(self):
        self.model_name_environment_variable = "OPENAI_COMPLETION_MODEL"
        self.endpoint_environment_variable = "OPENAI_COMPLETION_ENDPOINT"
        self.api_key_environment_variable = "OPENAI_COMPLETION_API_KEY"

    @limit_requests_per_minute
    @pyrit_target_retry
    async def send_prompt_async(self, *, prompt_request: PromptRequestResponse) -> PromptRequestResponse:

        self._validate_request(prompt_request=prompt_request)
        request_piece = prompt_request.request_pieces[0]

        logger.info(f"Sending the following prompt to the prompt target: {request_piece}")

        self.refresh_auth_headers()

        body = await self._construct_request_body(request=request_piece)

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
                return handle_bad_request_exception(response_text=StatusError.response.text, request=request_piece)
            elif StatusError.response.status_code == 429:
                raise RateLimitException()
            else:
                raise

        logger.info(f'Received the following response from the prompt target "{str_response.text}"')

        response_entry = self._construct_prompt_response_from_openai_json(
            open_ai_str_response=str_response.text, request_piece=request_piece
        )

        return response_entry

    async def _construct_request_body(self, request: PromptRequestPiece) -> dict:

        body_parameters = {
            "model": self._model_name,
            "prompt": request.converted_value,
            "top_p": self._top_p,
            "temperature": self._temperature,
            "frequency_penalty": self._frequency_penalty,
            "presence_penalty": self._presence_penalty,
            "max_tokens": self._max_tokens,
            "n": self._n,
        }

        # Filter out None values
        return {k: v for k, v in body_parameters.items() if v is not None}

    def _construct_prompt_response_from_openai_json(
        self,
        *,
        open_ai_str_response: str,
        request_piece: PromptRequestPiece,
    ) -> PromptRequestResponse:

        response = json.loads(open_ai_str_response)

        extracted_response = []
        for response_piece in response["choices"]:
            extracted_response.append(response_piece["text"])

        if not extracted_response:
            logger.log(logging.ERROR, "The chat returned an empty response.")
            raise EmptyResponseException(message="The chat returned an empty response.")

        return construct_response_from_request(request=request_piece, response_text_pieces=extracted_response)

    def _validate_request(self, *, prompt_request: PromptRequestResponse) -> None:
        if len(prompt_request.request_pieces) != 1:
            raise ValueError("This target only supports a single prompt request piece.")

        if prompt_request.request_pieces[0].converted_value_data_type != "text":
            raise ValueError("This target only supports text prompt input.")

        request = prompt_request.request_pieces[0]
        messages = self._memory.get_chat_messages_with_conversation_id(conversation_id=request.conversation_id)

        if len(messages) > 0:
            raise ValueError("This target only supports a single turn conversation.")

    def is_json_response_supported(self) -> bool:
        """Indicates that this target supports JSON response format."""
        return False
