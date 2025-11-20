# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging
from typing import Literal, Optional

from openai import BadRequestError, RateLimitError, APIStatusError

from pyrit.exceptions import (
    RateLimitException,
    handle_bad_request_exception,
    pyrit_target_retry,
)
from pyrit.models import (
    Message,
    MessagePiece,
    construct_response_from_request,
    data_serializer_factory,
)
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
        language: str = "en",
        speed: Optional[float] = None,
        **kwargs,
    ):
        """
        Initialize the TTS target with specified parameters.

        Args:
            model_name (str, Optional): The name of the model. Defaults to "tts-1".
            endpoint (str, Optional): The target URL for the OpenAI service.
            api_key (str, Optional): The API key for accessing the Azure OpenAI service.
                Defaults to the `OPENAI_TTS_KEY` environment variable.
            headers (str, Optional): Headers of the endpoint (JSON).
            use_entra_auth (bool, Optional): When set to True, user authentication is used
                instead of API Key. DefaultAzureCredential is taken for
                https://cognitiveservices.azure.com/.default . Please run `az login` locally
                to leverage user AuthN.
            max_requests_per_minute (int, Optional): Number of requests the target can handle per
                minute before hitting a rate limit. The number of requests sent to the target
                will be capped at the value provided.
            voice (str, Optional): The voice to use for TTS. Defaults to "alloy".
            response_format (str, Optional): The format of the audio response. Defaults to "mp3".
            language (str): The language for TTS. Defaults to "en".
            speed (float, Optional): The speed of the TTS. Select a value from 0.25 to 4.0. 1.0 is normal.
            httpx_client_kwargs (dict, Optional): Additional kwargs to be passed to the
                httpx.AsyncClient() constructor.
                For example, to specify a 3 minutes timeout: httpx_client_kwargs={"timeout": 180}
        """
        super().__init__(**kwargs)

        if not self._model_name:
            self._model_name = "tts-1"

        # Accept base URLs (/v1), specific API paths (/audio/speech), Azure formats
        tts_url_patterns = [r"/v1$", r"/audio/speech", r"/deployments/[^/]+/", r"openai/v1", r"\.models\.ai\.azure\.com"]
        self._warn_if_irregular_endpoint(tts_url_patterns)

        self._voice = voice
        self._response_format = response_format
        self._language = language
        self._speed = speed

    def _set_openai_env_configuration_vars(self):
        self.model_name_environment_variable = "OPENAI_TTS_MODEL"
        self.endpoint_environment_variable = "OPENAI_TTS_ENDPOINT"
        self.api_key_environment_variable = "OPENAI_TTS_KEY"

    @limit_requests_per_minute
    @pyrit_target_retry
    async def send_prompt_async(self, *, message: Message) -> Message:
        self._validate_request(message=message)
        request = message.message_pieces[0]

        logger.info(f"Sending the following prompt to the prompt target: {request}")

        # Construct request parameters for SDK
        body_parameters: dict[str, object] = {
            "model": self._model_name,
            "input": request.converted_value,
            "voice": self._voice,
            "response_format": self._response_format,
        }
        
        # Add optional parameters
        if self._speed is not None:
            body_parameters["speed"] = self._speed

        try:
            # SDK returns audio content directly
            audio_content = await self._async_client.audio.speech.create(**body_parameters)
            audio_bytes = audio_content.content
        except BadRequestError as bre:
            # Handle bad request (including content filter)
            return handle_bad_request_exception(
                response_text=bre.response.text if bre.response else str(bre),
                request=request,
                error_code=bre.status_code if hasattr(bre, 'status_code') else 400,
            )
        except RateLimitError:
            raise RateLimitException()
        except APIStatusError:
            raise

        logger.info("Received valid response from the prompt target")

        audio_response = data_serializer_factory(
            category="prompt-memory-entries", data_type="audio_path", extension=self._response_format
        )

        await audio_response.save_data(data=audio_bytes)

        response_entry = construct_response_from_request(
            request=request, response_text_pieces=[str(audio_response.value)], response_type="audio_path"
        )

        return response_entry

    def _validate_request(self, *, message: Message) -> None:
        n_pieces = len(message.message_pieces)
        if n_pieces != 1:
            raise ValueError("This target only supports a single message piece. " f"Received: {n_pieces} pieces.")

        piece_type = message.message_pieces[0].converted_value_data_type
        if piece_type != "text":
            raise ValueError(f"This target only supports text prompt input. Received: {piece_type}.")

        request = message.message_pieces[0]
        messages = self._memory.get_conversation(conversation_id=request.conversation_id)

        n_messages = len(messages)
        if n_messages > 0:
            raise ValueError(
                "This target only supports a single turn conversation. "
                f"Received: {n_messages} messages which indicates a prior turn."
            )

    def is_json_response_supported(self) -> bool:
        """Indicates that this target supports JSON response format."""
        return False
