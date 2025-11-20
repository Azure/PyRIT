# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging
from typing import Optional

from openai import BadRequestError, RateLimitError, APIStatusError

from pyrit.exceptions.exception_classes import (
    EmptyResponseException,
    RateLimitException,
    handle_bad_request_exception,
    pyrit_target_retry,
)
from pyrit.models import Message, MessagePiece, construct_response_from_request
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
        Initialize the OpenAICompletionTarget with the given parameters.

        Args:
            model_name (str, Optional): The name of the model.
            endpoint (str, Optional): The target URL for the OpenAI service.
            api_key (str, Optional): The API key for accessing the Azure OpenAI service.
                Defaults to the `OPENAI_CHAT_KEY` environment variable.
            headers (str, Optional): Headers of the endpoint (JSON).
            use_entra_auth (bool, Optional): When set to True, user authentication is used
                instead of API Key. DefaultAzureCredential is taken for
                https://cognitiveservices.azure.com/.default . Please run `az login` locally
                to leverage user AuthN.
            max_requests_per_minute (int, Optional): Number of requests the target can handle per
                minute before hitting a rate limit. The number of requests sent to the target
                will be capped at the value provided.
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
                `httpx.AsyncClient()` constructor.
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
    async def send_prompt_async(self, *, message: Message) -> Message:

        self._validate_request(message=message)
        message_piece = message.message_pieces[0]

        logger.info(f"Sending the following prompt to the prompt target: {message_piece}")

        # Build request parameters
        body_parameters = {
            "model": self._model_name,
            "prompt": message_piece.converted_value,
            "top_p": self._top_p,
            "temperature": self._temperature,
            "frequency_penalty": self._frequency_penalty,
            "presence_penalty": self._presence_penalty,
            "max_tokens": self._max_tokens,
            "n": self._n,
        }

        # Filter out None values
        request_params = {k: v for k, v in body_parameters.items() if v is not None}

        try:
            completion_response = await self._async_client.completions.create(**request_params)
        except BadRequestError as bre:
            # Handle bad request (including content filter)
            return handle_bad_request_exception(response_text=bre.response.text, request=message_piece)
        except RateLimitError:
            raise RateLimitException()
        except APIStatusError:
            raise

        logger.info(f'Received response from the prompt target with {len(completion_response.choices)} choices')

        # Extract response text from choices
        extracted_response = []
        for choice in completion_response.choices:
            extracted_response.append(choice.text)

        if not extracted_response:
            logger.log(logging.ERROR, "The completion returned an empty response.")
            raise EmptyResponseException(message="The completion returned an empty response.")

        return construct_response_from_request(request=message_piece, response_text_pieces=extracted_response)

    def _validate_request(self, *, message: Message) -> None:
        n_pieces = len(message.message_pieces)
        if n_pieces != 1:
            raise ValueError(f"This target only supports a single message piece. Received: {n_pieces} pieces.")

        piece_type = message.message_pieces[0].converted_value_data_type
        if piece_type != "text":
            raise ValueError(f"This target only supports text prompt input. Received: {piece_type}.")

    def is_json_response_supported(self) -> bool:
        """Indicates that this target supports JSON response format."""
        return False
