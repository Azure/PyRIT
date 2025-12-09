# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging
from typing import Any, Optional

from pyrit.exceptions.exception_classes import (
    pyrit_target_retry,
)
from pyrit.models import Message, construct_response_from_request
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
        self.underlying_model_environment_variable = "OPENAI_COMPLETION_UNDERLYING_MODEL"

    def _get_target_api_paths(self) -> list[str]:
        """Return API paths that should not be in the URL."""
        return ["/completions", "/v1/completions"]

    def _get_provider_examples(self) -> dict[str, str]:
        """Return provider-specific example URLs."""
        return {
            ".openai.azure.com": "https://{resource}.openai.azure.com/openai/v1",
            "api.openai.com": "https://api.openai.com/v1",
        }

    @limit_requests_per_minute
    @pyrit_target_retry
    async def send_prompt_async(self, *, message: Message) -> list[Message]:

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

        # Use unified error handler - automatically detects Completion and validates
        response = await self._handle_openai_request(
            api_call=lambda: self._async_client.completions.create(**request_params),  # type: ignore[call-overload]
            request=message,
        )
        return [response]

    async def _construct_message_from_response(self, response: Any, request: Any) -> Message:
        """
        Construct a Message from a Completion response.

        Args:
            response: The Completion response from OpenAI SDK.
            request: The original request MessagePiece.

        Returns:
            Message: Constructed message with extracted text.
        """
        logger.info(f"Received response from the prompt target with {len(response.choices)} choices")

        # Extract response text from validated choices
        extracted_response = [choice.text for choice in response.choices]

        return construct_response_from_request(request=request, response_text_pieces=extracted_response)

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

    async def _fetch_underlying_model_async(self) -> Optional[str]:
        """
        Fetch the underlying model name by making a minimal completion request.

        Sends a simple prompt with max_tokens=1 to minimize cost and latency,
        then extracts the model name from the response.

        Returns:
            Optional[str]: The underlying model name, or None if it cannot be determined.
        """
        try:
            response = await self._async_client.completions.create(
                model=self._model_name,
                prompt="hi",
                max_tokens=1,
            )

            raw_model = getattr(response, "model", None)
            return raw_model
        except Exception as e:
            logger.warning(f"Failed to fetch underlying model from endpoint: {e}")
            return None
