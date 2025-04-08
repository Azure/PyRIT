# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging
from typing import Literal, Optional

import httpx

from pyrit.common import net_utility
from pyrit.exceptions import (
    RateLimitException,
    handle_bad_request_exception,
    pyrit_target_retry,
)
from pyrit.models import (
    PromptRequestResponse,
    construct_response_from_request,
    data_serializer_factory,
)
from pyrit.models.prompt_request_piece import PromptRequestPiece
from pyrit.prompt_target import OpenAITarget, limit_requests_per_minute

logger = logging.getLogger(__name__)

TTSModel = Literal["tts-1", "tts-1-hd"]
TTSVoice = Literal["alloy", "echo", "fable", "onyx", "nova", "shimmer"]
TTSResponseFormat = Literal["flac", "mp3", "mp4", "mpeg", "mpga", "m4a", "ogg", "wav", "webm"]


class OpenAITTSTarget(OpenAITarget):

    def __init__(
        self,
        *,
        voice: TTSVoice = "alloy",
        response_format: TTSResponseFormat = "mp3",
        language: Optional[str] = "en",
        speed: Optional[float] = None,
        api_version: str = "2025-02-01-preview",
        **kwargs,
    ):
        """
        Initialize the TTS target with specified parameters.

        Args:
            model_name (str, Optional): The name of the model. Defaults to "tts-1".
            endpoint (str, Optional): The target URL for the OpenAI service.
            api_key (str, Optional): The API key for accessing the Azure OpenAI service.
                Defaults to the OPENAI_TTS_KEY environment variable.
            headers (str, Optional): Headers of the endpoint (JSON).
            use_aad_auth (bool, Optional): When set to True, user authentication is used
                instead of API Key. DefaultAzureCredential is taken for
                https://cognitiveservices.azure.com/.default . Please run `az login` locally
                to leverage user AuthN.
            api_version (str, Optional): The version of the Azure OpenAI API. Defaults to
                "2025-02-01-preview".
            max_requests_per_minute (int, Optional): Number of requests the target can handle per
                minute before hitting a rate limit. The number of requests sent to the target
                will be capped at the value provided.
            httpx_client_kwargs (dict, Optional): Additional kwargs to be passed to the
                httpx.AsyncClient() constructor.
            voice (str, Optional): The voice to use for TTS. Defaults to "alloy".
            response_format (str, Optional): The format of the audio response. Defaults to "mp3".
            language (str, Optional): The language for TTS. Defaults to "en".
            speed (float, Optional): The speed of the TTS. Select a value from 0.25 to 4.0. 1.0 is normal.
            httpx_client_kwargs (dict, Optional): Additional kwargs to be passed to the
                httpx.AsyncClient() constructor.
                For example, to specify a 3 minutes timeout: httpx_client_kwargs={"timeout": 180}
        """

        super().__init__(**kwargs)

        if not self._model_name:
            self._model_name = "tts-1"

        self._voice = voice
        self._response_format = response_format
        self._language = language
        self._speed = speed
        self._api_version = api_version

    def _set_openai_env_configuration_vars(self):
        self.model_name_environment_variable = "OPENAI_TTS_MODEL"
        self.endpoint_environment_variable = "OPENAI_TTS_ENDPOINT"
        self.api_key_environment_variable = "OPENAI_TTS_KEY"

    @limit_requests_per_minute
    @pyrit_target_retry
    async def send_prompt_async(self, *, prompt_request: PromptRequestResponse) -> PromptRequestResponse:
        self._validate_request(prompt_request=prompt_request)
        request = prompt_request.request_pieces[0]

        logger.info(f"Sending the following prompt to the prompt target: {request}")

        # Refresh auth headers if using AAD
        self.refresh_auth_headers()

        body = self._construct_request_body(request=request)

        params = {}
        if self._api_version is not None:
            params["api-version"] = self._api_version

        try:
            response = await net_utility.make_request_and_raise_if_error_async(
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
                return handle_bad_request_exception(response_text=StatusError.response.text, request=request)
            elif StatusError.response.status_code == 429:
                raise RateLimitException()
            else:
                raise

        logger.info("Received valid response from the prompt target")

        audio_response = data_serializer_factory(
            category="prompt-memory-entries", data_type="audio_path", extension=self._response_format
        )

        data = response.content

        await audio_response.save_data(data=data)

        response_entry = construct_response_from_request(
            request=request, response_text_pieces=[str(audio_response.value)], response_type="audio_path"
        )

        return response_entry

    def _construct_request_body(self, request: PromptRequestPiece) -> dict:

        body_parameters: dict[str, object] = {
            "model": self._model_name,
            "input": request.converted_value,
            "voice": self._voice,
            "file": self._response_format,
            "language": self._language,
            "speed": self._speed,
        }

        # Filter out None values
        return {k: v for k, v in body_parameters.items() if v is not None}

    def _validate_request(self, *, prompt_request: PromptRequestResponse) -> None:
        if len(prompt_request.request_pieces) != 1:
            raise ValueError("This target only supports a single prompt request piece.")

        if prompt_request.request_pieces[0].converted_value_data_type != "text":
            raise ValueError("This target only supports text prompt input.")

        request = prompt_request.request_pieces[0]
        messages = self._memory.get_conversation(conversation_id=request.conversation_id)

        if len(messages) > 0:
            raise ValueError("This target only supports a single turn conversation.")

    def is_json_response_supported(self) -> bool:
        """Indicates that this target supports JSON response format."""
        return False
