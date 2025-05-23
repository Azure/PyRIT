# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import asyncio
import base64
import json
import logging
import wave
from dataclasses import dataclass, field
from typing import List, Literal, Optional, Tuple
from urllib.parse import urlencode

import websockets

from pyrit.exceptions.exception_classes import ServerErrorException
from pyrit.models import PromptRequestResponse
from pyrit.models.data_type_serializer import data_serializer_factory
from pyrit.models.prompt_request_response import construct_response_from_request
from pyrit.prompt_target import OpenAITarget, limit_requests_per_minute

logger = logging.getLogger(__name__)

RealTimeVoice = Literal["alloy", "echo", "shimmer"]


@dataclass
class RealtimeTargetResult:
    """
    Represents the result of a Realtime API request, containing audio data and transcripts.

    Attributes:
        audio_bytes: Raw audio data returned by the API
        transcripts: List of text transcripts generated from the audio
    """

    audio_bytes: bytes = field(default_factory=lambda: b"")
    transcripts: List[str] = field(default_factory=list)

    def flatten_transcripts(self) -> str:
        """
        Flattens the list of transcripts into a single string.

        Returns:
            A single string containing all transcripts, separated by newlines.
        """
        return "\n".join(self.transcripts)


class RealtimeTarget(OpenAITarget):

    def __init__(
        self,
        *,
        api_version: str = "2024-10-01-preview",
        system_prompt: Optional[str] = None,
        voice: Optional[RealTimeVoice] = None,
        existing_convo: Optional[dict] = None,
        **kwargs,
    ) -> None:
        """
        RealtimeTarget class for Azure OpenAI Realtime API.
        Read more at https://learn.microsoft.com/en-us/azure/ai-services/openai/realtime-audio-reference
            and https://platform.openai.com/docs/guides/realtime-websocket
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
            api_version (str, Optional): The version of the Azure OpenAI API. Defaults to "2024-10-01-preview".
            system_prompt (str, Optional): The system prompt to use. Defaults to "You are a helpful AI assistant".
            voice (literal str, Optional): The voice to use. Defaults to None.
                the only supported voices by the AzureOpenAI Realtime API are "alloy", "echo", and "shimmer".
            existing_convo (dict[str, websockets.WebSocketClientProtocol], Optional): Existing conversations.
            httpx_client_kwargs (dict, Optional): Additional kwargs to be passed to the
                httpx.AsyncClient() constructor.
                For example, to specify a 3 minutes timeout: httpx_client_kwargs={"timeout": 180}
        """

        super().__init__(api_version=api_version, **kwargs)

        self.system_prompt = system_prompt or "You are a helpful AI assistant"
        self.voice = voice
        self._existing_conversation = existing_convo if existing_convo is not None else {}

    def _set_openai_env_configuration_vars(self):
        self.model_name_environment_variable = "OPENAI_REALTIME_MODEL"
        self.endpoint_environment_variable = "AZURE_OPENAI_REALTIME_ENDPOINT"
        self.api_key_environment_variable = "OPENAI_REALTIME_API_KEY"

    async def connect(self):
        """
        Connects to Realtime API Target using websockets.
        Returns the WebSocket connection.
        """

        logger.info(f"Connecting to WebSocket: {self._endpoint}")

        query_params = {
            "deployment": self._model_name,
            "OpenAI-Beta": "realtime=v1",
        }

        self._add_auth_param_to_query_params(query_params)

        if self._api_version is not None:
            query_params["api-version"] = self._api_version

        url = f"{self._endpoint}?{urlencode(query_params)}"

        websocket = await websockets.connect(url)
        logger.info("Successfully connected to AzureOpenAI Realtime API")
        return websocket

    def _add_auth_param_to_query_params(self, query_params: dict) -> None:
        """
        Adds the authentication parameter to the query parameters. This is how
        Realtime API works, it doesn't use the headers for auth.

        Args:
            query_params (dict): The query parameters.
        """
        if self._api_key:
            query_params["api-key"] = self._api_key

        if self._azure_auth:
            query_params["access_token"] = self._azure_auth.refresh_token()

    def _set_system_prompt_and_config_vars(self):

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
        websocket = self._get_websocket(conversation_id=conversation_id)
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

        convo_id = prompt_request.request_pieces[0].conversation_id
        if convo_id not in self._existing_conversation:
            websocket = await self.connect()
            self._existing_conversation[convo_id] = websocket

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
            output_audio_path, result = await self.send_audio_async(
                filename=request.converted_value, conversation_id=convo_id
            )

        elif response_type == "text":
            output_audio_path, result = await self.send_text_async(
                text=request.converted_value, conversation_id=convo_id
            )
        else:
            raise ValueError(f"Unsupported response type: {response_type}")

        text_response_piece = construct_response_from_request(
            request=request, response_text_pieces=[result.flatten_transcripts()], response_type="text"
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
        output_filename: Optional[str] = None,
    ) -> str:
        """
        Saves audio bytes to a WAV file.

        Args:
            audio_bytes (bytes): Audio bytes to save.
            num_channels (int): Number of audio channels. Defaults to 1 for the PCM16 format
            sample_width (int): Sample width in bytes. Defaults to 2 for the PCM16 format
            sample_rate (int): Sample rate in Hz. Defaults to 16000 Hz for the PCM16 format
            output_filename (str): Output filename. If None, a UUID filename will be used.

        Returns:
            str: The path to the saved audio file.
        """
        data = data_serializer_factory(category="prompt-memory-entries", data_type="audio_path")

        await data.save_formatted_audio(
            data=audio_bytes,
            output_filename=output_filename,
            num_channels=num_channels,
            sample_width=sample_width,
            sample_rate=sample_rate,
        )

        return data.value

    async def cleanup_target(self):
        """
        Disconnects from the WebSocket server to clean up, cleaning up all existing conversations.
        """
        for conversation_id, websocket in self._existing_conversation.items():
            if websocket:
                await websocket.close()
                logger.info(f"Disconnected from {self._endpoint} with conversation ID: {conversation_id}")
        self._existing_conversation = {}

    async def cleanup_conversation(self, conversation_id: str):
        """
        Disconnects from the WebSocket server for a specific conversation

        Args:
            conversation_id (str): The conversation ID to disconnect from.

        """
        websocket = self._existing_conversation.get(conversation_id)
        if websocket:
            await websocket.close()
            logger.info(f"Disconnected from {self._endpoint} with conversation ID: {conversation_id}")
            del self._existing_conversation[conversation_id]

    async def send_response_create(self, conversation_id: str):
        """
        Sends response.create message to the WebSocket server.

        Args:
            conversation_id (str): Conversation ID
        """
        await self.send_event(event={"type": "response.create"}, conversation_id=conversation_id)

    async def receive_events(self, conversation_id: str) -> RealtimeTargetResult:
        """
        Continuously receive events from the WebSocket server.

        Args:
            conversation_id: conversation ID

        Returns:
            List[Union[bytes, str]]: Collection of conversation messages with audio data
            at index 0 (bytes) and transcript at index 1 (str) if available

        Raises:
            ConnectionError: If WebSocket connection is not valid
            ValueError: If received event doesn't match expected structure
        """
        websocket = self._get_websocket(conversation_id=conversation_id)

        result = RealtimeTargetResult()

        try:
            async for message in websocket:
                event = json.loads(message)
                serialized_event = json.dumps(event, indent=2)
                msg_response_type = event.get("type")

                if not msg_response_type:
                    logger.warning(f"Received event without type field: {serialized_event}")
                    continue

                logger.debug(f"Processing event type: {msg_response_type}")

                if msg_response_type == "response.done":
                    RealtimeTarget._handle_response_done_event(event=event, result=result)
                    break

                elif msg_response_type == "error":
                    logger.error(f"Received 'error' event: {serialized_event}")
                    break

                elif msg_response_type == "response.audio.delta":
                    audio_data = RealtimeTarget._handle_audio_delta_event(event=event)
                    result.audio_bytes += audio_data

                elif msg_response_type == "response.audio.done":
                    logger.debug(f"Processing 'audio.done' event: {serialized_event}")

                else:
                    logger.debug(f"Unhandled event type '{msg_response_type}' for event {serialized_event}")

        except websockets.ConnectionClosed as e:
            logger.error(f"WebSocket connection closed for conversation {conversation_id}: {e}")
        except Exception as e:
            logger.error(f"An unexpected error occurred for conversation {conversation_id}: {e}")
            raise

        logger.debug(
            f"Completed receive_events with {len(result.transcripts)} transcripts "
            f"and {len(result.audio_bytes)} bytes of audio"
        )
        return result

    def _get_websocket(self, *, conversation_id: str):
        """
        Get and validate the WebSocket connection for a conversation.

        Args:
            conversation_id: The conversation ID

        Returns:
            The WebSocket connection

        Raises:
            ConnectionError: If WebSocket connection is not established
        """
        websocket = self._existing_conversation.get(conversation_id)
        if websocket is None:
            raise ConnectionError(f"WebSocket connection is not established for conversation {conversation_id}")
        return websocket

    @staticmethod
    def _handle_response_done_event(*, event: dict, result: RealtimeTargetResult) -> None:
        """
        Process a response.done event and extract the transcript.

        Args:
            event: The event data
            conversation_messages: List to add the transcript to

        Raises:
            ValueError: If event structure doesn't match expectations
        """
        logger.debug(f"Processing 'response.done' event: {json.dumps(event, indent=2)}")

        if "response" not in event:
            raise ValueError("Missing 'response' field in response.done event")

        response = event["response"]

        # Check for failed status
        status = response.get("status")
        if status == "failed":
            error_details = RealtimeTarget._extract_error_details(response=response)
            raise ServerErrorException(message=error_details)

        if "output" not in response:
            raise ValueError("Missing 'output' field in response")

        output = response["output"]
        if not output or not isinstance(output, list):
            raise ValueError(f"Empty or invalid 'output' array in response: {output}")

        content = output[0].get("content")
        if not content or not isinstance(content, list) or len(content) == 0:
            raise ValueError(f"Missing or invalid 'content' in: {output[0]}")

        if "transcript" not in content[0]:
            raise ValueError(f"Missing 'transcript' in: {content[0]}")

        transcript = content[0]["transcript"]
        result.transcripts.append(transcript)
        logger.debug(f"Added transcript to conversation messages: {transcript[:50]}...")

    @staticmethod
    def _extract_error_details(*, response: dict) -> str:
        """
        Extract error details from a failed response.

        Args:
            response: The response data

        Returns:
            A formatted error message
        """
        status_details = response.get("status_details", {})
        error = status_details.get("error", {})
        error_type = error.get("type", "unknown")
        error_message = error.get("message", "No error message provided")
        return f"[{error_type}] {error_message}"

    @staticmethod
    def _handle_audio_delta_event(*, event: dict) -> bytes:
        """
        Process a response.audio.delta event and extract audio data.

        Args:
            event: The event data

        Returns:
            Decoded audio data as bytes
        """
        logger.debug(f"Processing audio.delta event: {json.dumps(event, indent=2)}")

        if "delta" not in event:
            raise ValueError("Missing 'delta' field in audio delta event")

        audio_data = base64.b64decode(event["delta"])
        logger.debug(f"Decoded {len(audio_data)} bytes of audio data")
        return audio_data

    async def send_text_async(self, text: str, conversation_id: str) -> Tuple[str, RealtimeTargetResult]:
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

        result = await asyncio.wait_for(receive_tasks, timeout=30.0)  # Wait for all responses to be received

        if not result.audio_bytes:
            raise RuntimeError("No audio received from the server.")

        output_audio_path = await self.save_audio(audio_bytes=result.audio_bytes)
        return output_audio_path, result

    async def send_audio_async(self, filename: str, conversation_id: str) -> Tuple[str, RealtimeTargetResult]:
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
            logger.error(f"Error sending audio: {e}")
            raise

        event = {"type": "input_audio_buffer.commit"}
        await asyncio.sleep(0.1)
        await self.send_event(event, conversation_id=conversation_id)
        await self.send_response_create(conversation_id=conversation_id)  # Sends response.create message

        result = await asyncio.wait_for(receive_tasks, timeout=30.0)
        if not result.audio_bytes:
            raise RuntimeError("No audio received from the server.")

        output_audio_path = await self.save_audio(result.audio_bytes, num_channels, sample_width, frame_rate)
        return output_audio_path, result

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
