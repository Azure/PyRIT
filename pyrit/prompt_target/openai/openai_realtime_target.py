# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import asyncio
import base64
import json
import logging
from typing import Dict, Literal, Optional
import wave
import websockets

from pyrit.models import PromptRequestResponse
from pyrit.models.data_type_serializer import data_serializer_factory
from pyrit.models.prompt_request_piece import PromptRequestPiece
from pyrit.prompt_target import limit_requests_per_minute
from pyrit.prompt_target.openai.openai_target import OpenAITarget

logger = logging.getLogger(__name__)

RealTimeVoice = Literal["alloy", "echo", "shimmer"]


class RealtimeTarget(OpenAITarget):

    def __init__(
        self,
        api_version: str = "2024-10-01-preview",
        system_prompt: Optional[str] = "You are a helpful AI assistant",
        voice: Optional[RealTimeVoice] = None,
        *args,
        **kwargs,
    ) -> None:
        """
        RealtimeTarget class for Azure OpenAI Realtime API.
        Read more at https://learn.microsoft.com/en-us/azure/ai-services/openai/realtime-audio-reference
            and https://platform.openai.com/docs/guides/realtime-websocket
        Args:
            api_version (str, Optional): The version of the Azure OpenAI API. Defaults to "2024-10-01-preview".
            system_prompt (str, Optional): The system prompt to use. Defaults to "You are a helpful AI assistant".
            voice (literal str, Optional): The voice to use. Defaults to None.
                the only supported voices by the AzureOpenAI Realtime API are "alloy", "echo", and "shimmer".
            *args: Additional positional arguments to be passed.
            **kwargs: Additional keyword arguments to be passed.
        """

        if (kwargs.get("use_aad_auth") is not None) and (kwargs.get("use_aad_auth") is True):
            raise NotImplementedError("AAD authentication not implemented for TTSTarget yet.")

        super().__init__(api_version=api_version, *args, **kwargs)

        self.system_prompt = system_prompt
        self.voice = voice
        self.websocket = None

    def _set_azure_openai_env_configuration_vars(self):
        self.deployment_environment_variable = "AZURE_OPENAI_REALTIME_DEPLOYMENT"
        self.endpoint_uri_environment_variable = "AZURE_OPENAI_REALTIME_API_WEBSOCKET_URL"
        self.api_key_environment_variable = "AZURE_OPENAI_REALTIME_API_KEY"

    async def connect(self):
        # Connects to Realtime API Target using websockets.

        logger.info(f"Connecting to WebSocket: {self._endpoint}")
        headers = {"Authorization": f"Bearer {self._api_key}", "OpenAI-Beta": "realtime=v1"}

        websocket_url = f"{self._endpoint}/openai/realtime?api-version={self._api_version}"
        websocket_url = f"{websocket_url}&deployment={self._deployment_name}&api-key={self._api_key}"

        self.websocket = await websockets.connect(
            websocket_url,
            extra_headers=headers,
        )
        logger.info("Successfully connected to AzureOpenAI Realtime API")

    def _set_system_prompt_and_config_vars(self):
        """
        Sets the system prompt and configuration variables for the target.

        """

        session_config = {
            "modalities": ["audio", "text"],
            "instructions": self.system_prompt,
            "input_audio_format": "pcm16",
            "output_audio_format": "pcm16",
            "turn_detection": None,
        }

        if self.voice:
            session_config["voice"] = self.voice
        return session_config

    async def send_event(self, event):
        """
        Sends an event to the WebSocket server.

        Args:
            event: Event to send.

        """
        await self.websocket.send(json.dumps(event))
        logger.debug(f"Event sent - type: {event['type']}")

    async def send_config(self):
        # Sends the session configuration to the WebSocket server.

        config_variables = self._set_system_prompt_and_config_vars()

        await self.send_event({"type": "session.update", "session": config_variables})
        logger.info("Session set up")
        print("Session set up")

    @limit_requests_per_minute
    async def send_prompt_async(self, *, prompt_request: PromptRequestResponse) -> PromptRequestResponse:
        # Sends a prompt to the target and returns the response.

        # Validation function
        self._validate_request(prompt_request=prompt_request)

        await self.send_config()
        request = prompt_request.request_pieces[0]
        prompt = request.converted_value
        response_type = request.converted_value_data_type

        # Order of messages sent varies based on the data format of the prompt
        if response_type == "audio_path":
            with wave.open(prompt, "rb") as wav_file:

                # Read WAV parameters
                num_channels = wav_file.getnchannels()
                sample_width = wav_file.getsampwidth()  # Should be 2 bytes for PCM16
                frame_rate = wav_file.getframerate()
                num_frames = wav_file.getnframes()

                audio_content = wav_file.readframes(num_frames)

            receive_tasks = asyncio.create_task(self.receive_events(convo_max=1))

            await self.send_audio(audio=audio_content)
            events = await receive_tasks
            output_audio_path = await self.save_audio(events[0], num_channels, sample_width, frame_rate)

        elif response_type == "text":
            await self.send_response_create()
            receive_tasks = asyncio.create_task(self.receive_events(convo_max=1))
            await self.send_text(prompt)
            events = await receive_tasks
            output_audio_path = await self.save_audio(events[0])

        text_response_piece = PromptRequestPiece(
            original_value=events[1],
            original_value_data_type="text",
            converted_value=events[1],
            role="assistant",
            converted_value_data_type="text",
        )
        audio_response_piece = PromptRequestPiece(
            original_value=output_audio_path,
            original_value_data_type="audio_path",
            converted_value=output_audio_path,
            role="assistant",
            converted_value_data_type="audio_path",
        )

        response_entry = self.construct_response_from_request(
            request=request, response_pieces=[audio_response_piece, text_response_piece]
        )

        return response_entry

    def construct_response_from_request(
        self,
        request: PromptRequestPiece,
        response_pieces: list[PromptRequestPiece],
        prompt_metadata: Optional[Dict[str, str]] = None,
    ) -> PromptRequestResponse:

        # Constructs a response entry from a request.

        return PromptRequestResponse(
            request_pieces=[
                PromptRequestPiece(
                    role="assistant",
                    original_value=resp_piece.original_value,
                    conversation_id=request.conversation_id,
                    labels=request.labels,
                    prompt_target_identifier=request.prompt_target_identifier,
                    orchestrator_identifier=request.orchestrator_identifier,
                    original_value_data_type=resp_piece.original_value_data_type,
                    converted_value_data_type=resp_piece.converted_value_data_type,
                    converted_value=resp_piece.converted_value,
                    prompt_metadata=prompt_metadata,
                )
                for resp_piece in response_pieces
            ]
        )

    async def save_audio(
        self,
        audio_bytes: bytes,
        num_channels: int = 1,
        sample_width: int = 2,
        sample_rate: int = 16000,
        output_filename: str = None,
    ):
        """
        Saves audio bytes to a WAV file.

        Args:
            audio_bytes (bytes): Audio bytes to save.
            num_channels (int): Number of audio channels. Defaults to 1 for the PCM16 format
            sample_width (int): Sample width in bytes. Defaults to 2 for the PCM16 format
            sample_rate (int): Sample rate in Hz. Defaults to 16000 Hz for the PCM16 format
        """
        data = data_serializer_factory(category="prompt-memory-entries", data_type="audio_path")
        if not output_filename:
            filename = await data.get_data_filename()
            output_filename = str(filename)

        await data.save_formatted_audio(
            data=audio_bytes,
            output_filename=output_filename,
            num_channels=num_channels,
            sample_width=sample_width,
            sample_rate=sample_rate,
        )
        return output_filename

    async def disconnect(self):
        """
        Disconnects from the WebSocket server.
        """
        if self.websocket:
            await self.websocket.close()
            self.websocket = None
            logger.info(f"Disconnected from {self._endpoint}")

    async def send_response_create(self):
        """
        Sends response.create message to the WebSocket server.
        """
        await self.send_event({"type": "response.create"})

    async def receive_events(self, convo_max) -> list:
        """
        Continuously receive events from the WebSocket server.
        Args:
            convo_max: Maximum number of completed conversations or errors to receive before stopping.

        """
        # ctr = 0 #TODO: do we need for multiturn convo w orchestrator?
        # done = False

        if self.websocket is None:
            logger.error("WebSocket connection is not established")
            raise Exception("WebSocket connection is not established")

        audio_transcript = None
        audio_buffer = b""
        conversation_messages = []
        try:
            async for message in self.websocket:
                event = json.loads(message)
                msg_response_type = event.get("type")
                if msg_response_type:
                    if msg_response_type == "response.done":
                        logger.debug(f"event is: {json.dumps(event, indent=2)}")
                        audio_transcript = event["response"]["output"][0]["content"][0]["transcript"]
                        conversation_messages.append(audio_transcript)
                        # ctr += 1
                        break
                    elif msg_response_type == "error":
                        # ctr += 1
                        logger.error(f"Error, event is: {json.dumps(event, indent=2)}")
                        break
                    elif msg_response_type == "response.audio.delta":
                        # Append audio data to buffer
                        audio_data = base64.b64decode(event["delta"])
                        audio_buffer += audio_data
                        logger.debug("Audio data appended to buffer")
                    elif msg_response_type == "response.audio.done":
                        logger.debug(f"event is: {json.dumps(event, indent=2)}")
                        conversation_messages.append(audio_buffer)
                    else:
                        logger.debug(f"event is: {json.dumps(event, indent=2)}")

        except websockets.ConnectionClosed as e:
            logger.error(f"WebSocket connection closed: {e}")
        except Exception as e:
            logger.error(f"An unexpected error occurred: {e}")
        return conversation_messages

    async def send_text(self, text):
        """
        Sends text prompt to the WebSocket server.
        Args:
            text: prompt to send.
        """
        logger.info(f"Sending text message: {text}")
        event = {
            "type": "conversation.item.create",
            "item": {"type": "message", "role": "user", "content": [{"type": "input_text", "text": text}]},
        }
        await self.send_event(event)

    async def send_audio(self, audio):
        """
        Send an audio message to the WebSocket server.

        Args:
            audio: Audio message to send.
        """
        try:
            audio_base64 = base64.b64encode(audio).decode("utf-8")

            event = {"type": "input_audio_buffer.append", "audio": audio_base64}

            # await asyncio.sleep(0.1)
            await self.send_event(event)

        except Exception as e:
            logger.info(f"Error sending audio: {e}")
            return
        event = {"type": "input_audio_buffer.commit"}
        await asyncio.sleep(0.1)
        await self.send_event(event)
        await self.send_response_create()  # Sends response.create message

    def _validate_request(self, *, prompt_request: PromptRequestResponse) -> None:
        """Validates the structure and content of a prompt request for compatibility of this target.

        Args:
            prompt_request (PromptRequestResponse): The prompt request response object.

        Raises:
            ValueError: If more than two request pieces are provided.
            ValueError: If any of the request pieces have a data type other than 'text' or 'audio_path'.
        """

        # Check the number of request pieces
        if len(prompt_request.request_pieces) != 1:
            raise ValueError("This target only supports one request piece.")

        converted_prompt_data_types = [
            request_piece.converted_value_data_type for request_piece in prompt_request.request_pieces
        ]

        # Some models may not support all of these
        for prompt_data_type in converted_prompt_data_types:
            if prompt_data_type not in ["text", "audio_path"]:
                raise ValueError("This target only supports text and audio_path.")

    def is_json_response_supported(self) -> bool:
        """Indicates that this target supports JSON response format."""
        return False
