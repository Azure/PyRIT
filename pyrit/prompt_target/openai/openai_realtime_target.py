# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import asyncio
import base64
import json
import logging
import wave
from typing import Literal, Optional
from urllib.parse import urlencode

import websockets

from pyrit.models import PromptRequestResponse
from pyrit.models.data_type_serializer import data_serializer_factory
from pyrit.models.prompt_request_response import construct_response_from_request
from pyrit.prompt_target import limit_requests_per_minute
from pyrit.prompt_target import OpenAITarget

logger = logging.getLogger(__name__)

RealTimeVoice = Literal["alloy", "echo", "shimmer"]


class RealtimeTarget(OpenAITarget):

    def __init__(
        self,
        *,
        api_version: str = "2024-10-01-preview",
        system_prompt: Optional[str] = "You are a helpful AI assistant",
        voice: Optional[RealTimeVoice] = None,
        existing_convo: Optional[dict] = {},
        **kwargs,
    ) -> None:
        """
        RealtimeTarget class for Azure OpenAI Realtime API.
        Read more at https://learn.microsoft.com/en-us/azure/ai-services/openai/realtime-audio-reference
            and https://platform.openai.com/docs/guides/realtime-websocket
        Args:
            model_name (str, Optional): The name of the model.
            target_uri (str, Optional): The target URL for the OpenAI service.
            api_key (str, Optional): The API key for accessing the Azure OpenAI service.
                Defaults to the AZURE_OPENAI_CHAT_KEY environment variable.
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
            api_version (str, Optional): The version of the Azure OpenAI API. Defaults to "2024-10-01-preview".
            system_prompt (str, Optional): The system prompt to use. Defaults to "You are a helpful AI assistant".
            voice (literal str, Optional): The voice to use. Defaults to None.
                the only supported voices by the AzureOpenAI Realtime API are "alloy", "echo", and "shimmer".
            existing_convo (dict[str, websockets.WebSocketClientProtocol], Optional): Existing conversations.
        """

        if (kwargs.get("use_aad_auth") is not None) and (kwargs.get("use_aad_auth") is True):
            raise NotImplementedError("AAD authentication not implemented for Realtime yet.")

        super().__init__(api_version=api_version, **kwargs)

        self.system_prompt = system_prompt
        self.voice = voice
        self._existing_conversation = existing_convo

    def _set_openai_env_configuration_vars(self):
        self.model_name_environment_variable = "AZURE_OPENAI_REALTIME_DEPLOYMENT"
        self.target_uri_environment_variable = "AZURE_OPENAI_REALTIME_API_WEBSOCKET_URL"
        self.api_key_environment_variable = "AZURE_OPENAI_REALTIME_API_KEY"

    async def connect(self):
        """
        Connects to Realtime API Target using websockets.
        Returns the WebSocket connection.
        """

        logger.info(f"Connecting to WebSocket: {self._target_uri}")

        query_params = {
            "api-version": self._api_version,
            "deployment": self._model_name,
            "api-key": self._api_key,
            "OpenAI-Beta": "realtime=v1",
        }
        url = f"{self._target_uri}?{urlencode(query_params)}"

        websocket = await websockets.connect(url)
        logger.info("Successfully connected to AzureOpenAI Realtime API")
        return websocket

    def _set_system_prompt_and_config_vars(self):
        # Sets the system prompt and configuration variables for the target.

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

    async def send_event(self, event: dict, conversation_id: str):
        """
        Sends an event to the WebSocket server.

        Args:
            event (dict): Event to send in dictionary format.
            conversation_id (str): Conversation ID
        """
        websocket = self._existing_conversation.get(conversation_id)

        if websocket is None:
            logger.error("WebSocket connection is not established")
            raise Exception("WebSocket connection is not established")
        await websocket.send(json.dumps(event))
        logger.debug(f"Event sent - type: {event['type']}")

    async def send_config(self, conversation_id: str):
        """
        Sends the session configuration to the WebSocket server.

        Args:
            conversation_id (str): Conversation ID
        """

        config_variables = self._set_system_prompt_and_config_vars()

        await self.send_event(
            event={"type": "session.update", "session": config_variables}, conversation_id=conversation_id
        )
        logger.info("Session set up")

    @limit_requests_per_minute
    async def send_prompt_async(self, *, prompt_request: PromptRequestResponse) -> PromptRequestResponse:
        # Sends a prompt to the target and returns the response.

        convo_id = prompt_request.request_pieces[0].conversation_id
        if convo_id not in self._existing_conversation:
            websocket = await self.connect()
            self._existing_conversation[convo_id] = websocket

            # Store system prompt in memory:
            self.set_system_prompt(
                system_prompt=self.system_prompt,
                conversation_id=convo_id,
                orchestrator_identifier=self.get_identifier(),
            )

        websocket = self._existing_conversation[convo_id]

        self._validate_request(prompt_request=prompt_request)

        await self.send_config(conversation_id=convo_id)
        request = prompt_request.request_pieces[0]
        response_type = request.converted_value_data_type

        # Order of messages sent varies based on the data format of the prompt
        if response_type == "audio_path":
            output_audio_path, events = await self.send_audio_async(
                filename=request.converted_value, conversation_id=convo_id
            )

        elif response_type == "text":
            output_audio_path, events = await self.send_text_async(
                text=request.converted_value, conversation_id=convo_id
            )

        text_response_piece = construct_response_from_request(
            request=request, response_text_pieces=[events[1]], response_type="text"
        ).request_pieces[0]

        audio_response_piece = construct_response_from_request(
            request=request, response_text_pieces=[output_audio_path], response_type="audio_path"
        ).request_pieces[0]

        response_entry = PromptRequestResponse(request_pieces=[text_response_piece, audio_response_piece])
        return response_entry

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

    async def cleanup_target(self):
        """
        Disconnects from the WebSocket server to clean up, cleaning up all existing conversations.
        """
        for conversation_id, websocket in self._existing_conversation.items():
            if websocket:
                await websocket.close()
                logger.info(f"Disconnected from {self._target_uri} with conversation ID: {conversation_id}")
        self._existing_conversation = {}

    async def cleanup_conversation(self, conversation_id: str):
        """
        Disconnects from the WebSocket server for a specific conversation
        """
        websocket = self._existing_conversation.get(conversation_id)
        if websocket:
            await websocket.close()
            logger.info(f"Disconnected from {self._target_uri} with conversation ID: {conversation_id}")
            del self._existing_conversation[conversation_id]

    async def send_response_create(self, conversation_id: str):
        """
        Sends response.create message to the WebSocket server.
        """
        await self.send_event(event={"type": "response.create"}, conversation_id=conversation_id)

    async def receive_events(self, conversation_id: str) -> list:
        """
        Continuously receive events from the WebSocket server.
        Args:
            conversation_id: conversation ID
        """
        websocket = self._existing_conversation[conversation_id]
        if websocket is None:  # change this to existing_conversation.websocket
            logger.error("WebSocket connection is not established")
            raise Exception("WebSocket connection is not established")

        audio_transcript = None
        audio_buffer = b""
        conversation_messages = []
        try:
            async for message in websocket:
                event = json.loads(message)
                msg_response_type = event.get("type")
                if msg_response_type:
                    if msg_response_type == "response.done":
                        logger.debug(f"event is: {json.dumps(event, indent=2)}")
                        audio_transcript = event["response"]["output"][0]["content"][0]["transcript"]
                        conversation_messages.append(audio_transcript)
                        break
                    elif msg_response_type == "error":
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

    async def send_text_async(self, text: str, conversation_id: str):
        """
        Sends text prompt to the WebSocket server.
        Args:
            text: prompt to send.
            conversation_id: conversation ID
        """
        await self.send_response_create(conversation_id=conversation_id)

        # Listen for responses
        receive_tasks = asyncio.create_task(self.receive_events(conversation_id=conversation_id))

        logger.info(f"Sending text message: {text}")
        event = {
            "type": "conversation.item.create",
            "item": {"type": "message", "role": "user", "content": [{"type": "input_text", "text": text}]},
        }
        await self.send_event(event=event, conversation_id=conversation_id)

        events = await receive_tasks  # Wait for all responses to be received

        output_audio_path = await self.save_audio(events[0])
        return output_audio_path, events

    async def send_audio_async(self, filename: str, conversation_id: str):
        """
        Send an audio message to the WebSocket server.

        Args:
            filename (str): The path to the audio file.
        """
        with wave.open(filename, "rb") as wav_file:

            # Read WAV parameters
            num_channels = wav_file.getnchannels()
            sample_width = wav_file.getsampwidth()  # Should be 2 bytes for PCM16
            frame_rate = wav_file.getframerate()
            num_frames = wav_file.getnframes()

            audio_content = wav_file.readframes(num_frames)

        receive_tasks = asyncio.create_task(self.receive_events(conversation_id=conversation_id))

        try:
            audio_base64 = base64.b64encode(audio_content).decode("utf-8")

            event = {"type": "input_audio_buffer.append", "audio": audio_base64}

            # await asyncio.sleep(0.1)
            await self.send_event(event=event, conversation_id=conversation_id)

        except Exception as e:
            logger.info(f"Error sending audio: {e}")
            return
        event = {"type": "input_audio_buffer.commit"}
        await asyncio.sleep(0.1)
        await self.send_event(event, conversation_id=conversation_id)
        await self.send_response_create(conversation_id=conversation_id)  # Sends response.create message

        responses = await receive_tasks
        output_audio_path = await self.save_audio(responses[0], num_channels, sample_width, frame_rate)
        return output_audio_path, responses

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

        if prompt_request.request_pieces[0].converted_value_data_type not in ["text", "audio_path"]:
            raise ValueError("This target only supports text and audio_path prompt input.")

    def is_json_response_supported(self) -> bool:
        """Indicates that this target supports JSON response format."""
        return False
