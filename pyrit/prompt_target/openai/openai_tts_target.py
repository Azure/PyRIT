# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging
from httpx import HTTPStatusError
from typing import Literal

from pyrit.common import net_utility
from pyrit.exceptions import RateLimitException
from pyrit.exceptions import handle_bad_request_exception
from pyrit.models import PromptRequestResponse
from pyrit.models import data_serializer_factory, construct_response_from_request
from pyrit.prompt_target import OpenAITarget, limit_requests_per_minute


logger = logging.getLogger(__name__)

TTSModel = Literal["tts-1", "tts-1-hd"]
TTSVoice = Literal["alloy", "echo", "fable", "onyx", "nova", "shimmer"]
TTSResponseFormat = Literal["flac", "mp3", "mp4", "mpeg", "mpga", "m4a", "ogg", "wav", "webm"]


class OpenAITTSTarget(OpenAITarget):

    def __init__(
        self,
        voice: TTSVoice = "alloy",
        response_format: TTSResponseFormat = "mp3",
        model: TTSModel = "tts-1",
        language: str = "en",
        api_version: str = "2024-03-01-preview",
        *args,
        **kwargs,
    ):

        if (kwargs.get("use_aad_auth") is not None) and (kwargs.get("use_aad_auth") is True):
            raise NotImplementedError("AAD authentication not implemented for TTSTarget yet.")

        super().__init__(*args, **kwargs)

        self._voice = voice
        self._model = model
        self._response_format = response_format
        self._language = language
        self._api_version = api_version

    def _set_azure_openai_env_configuration_vars(self):
        self.deployment_environment_variable = "AZURE_OPENAI_TTS_DEPLOYMENT"
        self.endpoint_uri_environment_variable = "AZURE_OPENAI_TTS_ENDPOINT"
        self.api_key_environment_variable = "AZURE_OPENAI_TTS_KEY"

    @limit_requests_per_minute
    async def send_prompt_async(self, *, prompt_request: PromptRequestResponse) -> PromptRequestResponse:
        self._validate_request(prompt_request=prompt_request)
        request = prompt_request.request_pieces[0]

        logger.info(f"Sending the following prompt to the prompt target: {request}")

        body: dict[str, object] = {
            "model": self._model,
            "input": request.converted_value,
            "voice": self._voice,
            "file": self._response_format,
            "language": self._language,
        }

        self._extra_headers["api-key"] = self._api_key

        response_entry = None
        try:
            # Note the openai client doesn't work here, potentially due to a mismatch
            response = await net_utility.make_request_and_raise_if_error_async(
                endpoint_uri=f"{self._endpoint}/openai/deployments/{self._deployment_name}/"
                f"audio/speech?api-version={self._api_version}",
                method="POST",
                headers=self._extra_headers,
                request_body=body,
            )
        except HTTPStatusError as hse:
            if hse.response.status_code == 400:
                # Handle Bad Request
                response_entry = handle_bad_request_exception(response_text=hse.response.text, request=request)
            elif hse.response.status_code == 429:
                raise RateLimitException()
            else:
                raise hse

        logger.info("Received valid response from the prompt target")

        audio_response = data_serializer_factory(data_type="audio_path", extension=self._response_format)

        data = response.content

        await audio_response.save_data(data=data)

        if not response_entry:
            response_entry = construct_response_from_request(
                request=request, response_text_pieces=[str(audio_response.value)], response_type="audio_path"
            )

        return response_entry

    def _validate_request(self, *, prompt_request: PromptRequestResponse) -> None:
        if len(prompt_request.request_pieces) != 1:
            raise ValueError("This target only supports a single prompt request piece.")

        if prompt_request.request_pieces[0].converted_value_data_type != "text":
            raise ValueError("This target only supports text prompt input.")

        request = prompt_request.request_pieces[0]
        messages = self._memory.get_chat_messages_with_conversation_id(conversation_id=request.conversation_id)

        if len(messages) > 0:
            raise ValueError("This target only supports a single turn conversation.")
