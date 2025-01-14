# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import asyncio
import base64
import json
import logging
from typing import Dict, Optional
import wave
import websockets

from pyrit.common import default_values
from pyrit.models import PromptRequestResponse
from pyrit.models.prompt_request_piece import PromptRequestPiece
from pyrit.prompt_target import PromptTarget, limit_requests_per_minute

logger = logging.getLogger(__name__)


class RealtimeTarget(PromptTarget):
    REALTIME_ENDPOINT_WEBSOCKET_URL = "AZURE_OPENAI_REALTIME_API_WS_URL"
    REALTIME_DEPLOYMENT = "AZURE_OPENAI_REALTIME_DEPLOYMENT"
    REALTIME_API_KEY = "AZURE_OPENAI_REALTIME_API_KEY"
    REALTIME_API_VERSION = "AZURE_OPENAI_REALTIME_API_VERSION"

    def __init__(
        self,
        key: str = None,
        deployment: str = None,
        api_version: str = None,
        url: str = None,
        *args,
        **kwargs,
    ) -> None:

        self.api_key = default_values.get_required_value(env_var_name=self.REALTIME_API_KEY, passed_value=key)
        self.url = default_values.get_required_value(
            env_var_name=self.REALTIME_ENDPOINT_WEBSOCKET_URL, passed_value=url
        )
        self.deployment = default_values.get_required_value(
            env_var_name=self.REALTIME_DEPLOYMENT, passed_value=deployment
        )
        self.api_version = default_values.get_required_value(
            env_var_name=self.REALTIME_API_VERSION, passed_value=api_version
        )

        self.websocket = None
        super().__init__(*args, **kwargs)

    async def connect(self):
        """
        Connects to the WebSocket server.

        """
        logger.info(f"Connecting to WebSocket: {self.url}")
        headers = {"Authorization": f"Bearer {self.api_key}", "OpenAI-Beta": "realtime=v1"}

        url = f"{self.url}/openai/realtime?api-version={self.api_version}"
        url = f"{url}&deployment={self.deployment}&api-key={self.api_key}"

        self.websocket = await websockets.connect(
            url,
            extra_headers=headers,
        )
        logger.info("Successfully connected to AzureOpenAI Realtime API")

    @limit_requests_per_minute
    async def send_prompt_async(self, *, prompt_request: PromptRequestResponse) -> PromptRequestResponse:
        # Validation function
        self._validate_request(prompt_request=prompt_request)

        request = prompt_request.request_pieces[0]
        prompt = request.converted_value

        await self.send_config()

        response_type = request.converted_value_data_type
        if response_type == "audio_path":
            with wave.open(prompt, "rb") as wav_file:
                # Read WAV parameters and add small silence to audio
                num_channels = wav_file.getnchannels()

                sample_width = wav_file.getsampwidth()  # Should be 2 bytes for PCM16
                frame_rate = wav_file.getframerate()
                num_frames = wav_file.getnframes()

                silence_duration = 0.5  # 500ms
                silence_frames = int(frame_rate * silence_duration)
                silence = (b"\x00" * sample_width) * silence_frames * num_channels

                wav_data = wav_file.readframes(num_frames)

                audio_content = wav_data + silence

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
            # converted_value=events[1],
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
        prompt_metadata: Optional[Dict[str,str]] = None,
    ) -> PromptRequestResponse:
        """
        Constructs a response entry from a request.
        """
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
        self, audio_bytes: bytes, num_channels: int = 1, sample_width: int = 2, sample_rate: int = 16000
    ):
        """
        Saves audio bytes to a WAV file.

        Args:
            audio_bytes (bytes): Audio bytes to save.
            num_channels (int): Number of audio channels. Defaults to 1 for the PCM16 format
            sample_width (int): Sample width in bytes. Defaults to 2 for the PCM16 format
            sample_rate (int): Sample rate in Hz. Defaults to 16000 Hz for the PCM16 format
        """
        output_filename = "response_audio.wav"

        with wave.open(output_filename, "wb") as wav_file:
            wav_file.setnchannels(num_channels)
            wav_file.setsampwidth(sample_width)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(audio_bytes)

        return output_filename

    async def disconnect(self):
        """
        Disconnects from the WebSocket server.
        """
        if self.websocket:
            await self.websocket.close()
            self.websocket = None
            logger.info(f"Disconnected from {self.url}")

    async def send_event(self, event):
        """
        Sends event to the WebSocket server.
        args:
            event: Event data to send (from the user)
        """
        await self.websocket.send(json.dumps(event))
        logger.debug(f"Event sent - type: {event['type']}")

    async def send_config(self):
        instructions = "You are a helpful AI"
        session_config = {
            "modalities": ["audio", "text"],
            "instructions": instructions,
            # "voice": self.voice,
            "input_audio_format": "pcm16",
            "output_audio_format": "pcm16",
            "turn_detection": None,
            # "temperature": 0.6
        }

        await self.send_event({"type": "session.update", "session": session_config})
        logger.info("Session set up")
        print("Session set up")

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
        ctr = 0
        done = False

        if self.websocket is None:
            logger.error("WebSocket connection is not established")
            raise Exception("WebSocket connection is not established")

        while not done:
            audio_transcript = None
            audio_buffer = b""
            conversation_messages = []
            try:
                async for message in self.websocket:
                    event = json.loads(message)
                    msg_response_type = event["type"]
                    if msg_response_type:
                        if msg_response_type == "response.done":
                            logger.debug(f"event is: {json.dumps(event, indent=2)}")
                            audio_transcript = event["response"]["output"][0]["content"][0]["transcript"]
                            conversation_messages.append(audio_transcript)
                            ctr += 1
                            break
                        elif msg_response_type == "error":
                            ctr += 1
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

                if ctr >= convo_max:
                    done = True
                    break
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
            raise ValueError(
                "This target only supports one request piece."
            )  # TODO: should it support multiple? check this

        converted_prompt_data_types = [
            request_piece.converted_value_data_type for request_piece in prompt_request.request_pieces
        ]

        # Some models may not support all of these
        for prompt_data_type in converted_prompt_data_types:
            if prompt_data_type not in ["text", "audio_path"]:
                raise ValueError("This target only supports text and audio_path.")
