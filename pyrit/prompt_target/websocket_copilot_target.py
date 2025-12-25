# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import asyncio
import json
import logging
import os
import uuid
from enum import IntEnum
from typing import Optional

import websockets

from pyrit.exceptions import (
    EmptyResponseException,
    pyrit_target_retry,
)
from pyrit.models import Message, construct_response_from_request
from pyrit.prompt_target import PromptTarget, limit_requests_per_minute

logger = logging.getLogger(__name__)

# Useful links:
# https://github.com/mbrg/power-pwn/blob/main/src/powerpwn/copilot/copilot_connector/copilot_connector.py
# https://labs.zenity.io/p/access-copilot-m365-terminal


class CopilotMessageType(IntEnum):
    """Enumeration for Copilot WebSocket message types."""

    UNKNOWN = -1
    PARTIAL_RESPONSE = 1
    FINAL_CONTENT = 2
    STREAM_END = 3
    USER_PROMPT = 4
    PING = 6


class WebSocketCopilotTarget(PromptTarget):
    """
    A WebSocket-based prompt target for Microsoft Copilot integration.

    This target enables communication with Microsoft Copilot through a WebSocket connection.
    Currently, authentication requires manually extracting a WebSocket URL from an active browser session.
    In the future, more flexible authentication mechanisms will be added.

    To obtain the WebSocket URL:
        1. Ensure you are logged into Microsoft 365 with access to Copilot
        2. Navigate to https://m365.cloud.microsoft/chat or open Copilot in https://teams.microsoft.com/v2
        3. Open browser developer tools and switch to the Network tab
        4. Begin typing or send a message to Copilot to establish the WebSocket connection
        5. Search the network requests for "chathub", "conversation", or "access_token"
        6. Identify the WebSocket connection (look for WS protocol) and copy its full URL

    Warning:
        All target instances using the same `WEBSOCKET_URL` will share a single conversation session.
        Only works with licensed Microsoft 365 Copilot. The free Copilot version is not compatible.
    """

    # TODO: add more flexible auth, use puppeteer? https://github.com/mbrg/power-pwn/blob/main/src/powerpwn/copilot/copilot_connector/copilot_connector.py#L248

    SUPPORTED_DATA_TYPES = {"text"}  # TODO: support more types?

    RESPONSE_TIMEOUT_SECONDS: int = 60
    CONNECTION_TIMEOUT_SECONDS: int = 30

    def __init__(
        self,
        *,
        verbose: bool = False,
        max_requests_per_minute: Optional[int] = None,
        model_name: str = "copilot",
        response_timeout_seconds: int = RESPONSE_TIMEOUT_SECONDS,
    ) -> None:
        """
        Initialize the WebSocketCopilotTarget.

        Args:
            verbose (bool): Enable verbose logging. Defaults to False.
            max_requests_per_minute (int, Optional): Maximum number of requests per minute.
            model_name (str): The model name. Defaults to "copilot".
            response_timeout_seconds (int): Timeout for receiving responses in seconds. Defaults to 60s.

        Raises:
            ValueError: If WebSocket URL is not provided, is empty, or has invalid format.
            ValueError: If required parameters are missing or empty in the WebSocket URL.
        """
        self._websocket_url = os.getenv("WEBSOCKET_URL")
        if not self._websocket_url or self._websocket_url.strip() == "":
            raise ValueError("WebSocket URL must be provided through the WEBSOCKET_URL environment variable")

        if not self._websocket_url.startswith("wss://"):
            raise ValueError(f"WebSocket URL must start with 'wss://'. Received: {self._websocket_url[:10]}")

        if "ConversationId=" not in self._websocket_url:
            raise ValueError("`ConversationId` parameter not found in WebSocket URL.")
        self._conversation_id = self._websocket_url.split("ConversationId=")[1].split("&")[0]
        if not self._conversation_id:
            raise ValueError("`ConversationId` parameter is empty in WebSocket URL.")

        if "X-SessionId=" not in self._websocket_url:
            raise ValueError("`X-SessionId` parameter not found in WebSocket URL.")
        self._session_id = self._websocket_url.split("X-SessionId=")[1].split("&")[0]
        if not self._session_id:
            raise ValueError("`X-SessionId` parameter is empty in WebSocket URL.")

        super().__init__(
            verbose=verbose,
            max_requests_per_minute=max_requests_per_minute,
            endpoint=self._websocket_url.split("?")[0],  # wss://substrate.office.com/m365Copilot/Chathub/...
            model_name=model_name,
        )

        if response_timeout_seconds <= 0:
            raise ValueError("response_timeout_seconds must be a positive integer.")
        self._response_timeout_seconds = response_timeout_seconds

        if self._verbose:
            logger.info(f"WebSocketCopilotTarget initialized with conversation_id: {self._conversation_id}")
            logger.info(f"Session ID: {self._session_id}")

    @staticmethod
    def _dict_to_websocket(data: dict) -> str:
        # Produce the smallest possible JSON string, followed by record separator
        return json.dumps(data, separators=(",", ":")) + "\x1e"

    @staticmethod
    def _parse_raw_message(message: str) -> list[tuple[CopilotMessageType, str]]:
        """
        Extract actionable content from a raw WebSocket message.
        Returns more than one JSON message if multiple are found.

        Args:
            message (str): The raw WebSocket message string.

        Returns:
            list[tuple[CopilotMessageType, str]]: A list of tuples where each tuple contains
                message type and extracted content.
        """
        results: list[tuple[CopilotMessageType, str]] = []

        # https://github.com/dotnet/aspnetcore/blob/main/src/SignalR/docs/specs/HubProtocol.md#json-encoding
        messages = message.split("\x1e")  # record separator

        for message in messages:
            if not message or not message.strip():
                continue

            try:
                data = json.loads(message)
                msg_type = CopilotMessageType(data.get("type", -1))

                if msg_type in (
                    CopilotMessageType.PING,
                    CopilotMessageType.PARTIAL_RESPONSE,
                    CopilotMessageType.STREAM_END,
                ):
                    results.append((msg_type, ""))
                    continue

                if msg_type == CopilotMessageType.FINAL_CONTENT:
                    bot_text = data.get("item", {}).get("result", {}).get("message", "")
                    if not bot_text:
                        # In this case, EmptyResponseException will be raised anyway
                        logger.warning("FINAL_CONTENT received but no parseable content found.")
                    results.append((CopilotMessageType.FINAL_CONTENT, bot_text))
                    continue

                results.append((msg_type, ""))

            except json.JSONDecodeError as e:
                logger.error(f"Failed to decode JSON message: {str(e)}")
                results.append((CopilotMessageType.UNKNOWN, ""))

        return results if results else [(CopilotMessageType.UNKNOWN, "")]

    def _build_prompt_message(self, prompt: str) -> dict:
        return {
            "arguments": [
                {
                    "source": "officeweb",  # TODO: support 'teamshub' as well
                    # TODO: not sure whether to uuid.uuid4() or use a static like it's done in power-pwn
                    # https://github.com/mbrg/power-pwn/blob/main/src/powerpwn/copilot/copilot_connector/copilot_connector.py#L156
                    "clientCorrelationId": str(uuid.uuid4()),
                    "sessionId": self._session_id,
                    "optionsSets": [
                        "enterprise_flux_web",
                        "enterprise_flux_work",
                        "enable_request_response_interstitials",
                        "enterprise_flux_image_v1",
                        "enterprise_toolbox_with_skdsstore",
                        "enterprise_toolbox_with_skdsstore_search_message_extensions",
                        "enable_ME_auth_interstitial",
                        "skdsstorethirdparty",
                        "enable_confirmation_interstitial",
                        "enable_plugin_auth_interstitial",
                        "enable_response_action_processing",
                        "enterprise_flux_work_gptv",
                        "enterprise_flux_work_code_interpreter",
                        "enable_batch_token_processing",
                    ],
                    "options": {},
                    "allowedMessageTypes": [
                        "Chat",
                        "Suggestion",
                        "InternalSearchQuery",
                        "InternalSearchResult",
                        "Disengaged",
                        "InternalLoaderMessage",
                        "RenderCardRequest",
                        "AdsQuery",
                        "SemanticSerp",
                        "GenerateContentQuery",
                        "SearchQuery",
                        "ConfirmationCard",
                        "AuthError",
                        "DeveloperLogs",
                    ],
                    "sliceIds": [],
                    # TODO: enable using agents https://github.com/mbrg/power-pwn/blob/main/src/powerpwn/copilot/copilot_connector/copilot_connector.py#L192
                    "threadLevelGptId": {},
                    "conversationId": self._conversation_id,
                    "traceId": str(uuid.uuid4()).replace("-", ""),  # TODO: same case as clientCorrelationId
                    "isStartOfSession": 0,
                    "productThreadType": "Office",
                    "clientInfo": {"clientPlatform": "web"},
                    "message": {
                        "author": "user",
                        "inputMethod": "Keyboard",
                        "text": prompt,
                        "entityAnnotationTypes": ["People", "File", "Event", "Email", "TeamsMessage"],
                        "requestId": str(uuid.uuid4()).replace("-", ""),
                        "locationInfo": {"timeZoneOffset": 0, "timeZone": "UTC"},
                        "locale": "en-US",
                        "messageType": "Chat",
                        "experienceType": "Default",
                    },
                    "plugins": [],  # TODO: support enabling some plugins?
                }
            ],
            "invocationId": "0",  # TODO: should be dynamic?
            "target": "chat",
            "type": CopilotMessageType.USER_PROMPT,
        }

    async def _connect_and_send(self, prompt: str) -> str:
        protocol_msg = {"protocol": "json", "version": 1}
        prompt_dict = self._build_prompt_message(prompt)

        inputs = [protocol_msg, prompt_dict]
        last_response = ""

        async with websockets.connect(
            self._websocket_url,
            open_timeout=self.CONNECTION_TIMEOUT_SECONDS,
            close_timeout=self.CONNECTION_TIMEOUT_SECONDS,
        ) as websocket:
            for input_msg in inputs:
                payload = self._dict_to_websocket(input_msg)
                await websocket.send(payload)

                stop_polling = False
                while not stop_polling:
                    try:
                        response = await asyncio.wait_for(
                            websocket.recv(),
                            timeout=self._response_timeout_seconds,
                        )
                    except asyncio.TimeoutError:
                        raise TimeoutError(
                            f"Timed out waiting for Copilot response after {self._response_timeout_seconds} seconds."
                        )

                    if response is None:
                        raise RuntimeError(
                            "WebSocket connection closed unexpectedly: received None from websocket.recv()"
                        )

                    parsed_messages = self._parse_raw_message(response)

                    for msg_type, content in parsed_messages:
                        if msg_type in (
                            CopilotMessageType.UNKNOWN,
                            CopilotMessageType.FINAL_CONTENT,
                            CopilotMessageType.STREAM_END,
                        ):
                            stop_polling = True

                            if msg_type == CopilotMessageType.FINAL_CONTENT:
                                last_response = content
                            elif msg_type == CopilotMessageType.UNKNOWN:
                                logger.debug("Received unknown or empty message type.")

            return last_response

    def _validate_request(self, *, message: Message) -> None:
        n_pieces = len(message.message_pieces)
        if n_pieces != 1:
            raise ValueError(f"This target only supports a single message piece. Received: {n_pieces} pieces.")

        piece_type = message.message_pieces[0].converted_value_data_type
        if piece_type != "text":
            raise ValueError(f"This target only supports text prompt input. Received: {piece_type}.")

    @limit_requests_per_minute
    @pyrit_target_retry
    async def send_prompt_async(self, *, message: Message) -> list[Message]:
        """
        Asynchronously send a message to Microsoft Copilot using WebSocket.

        Args:
            message (Message): A message to be sent to the target.

        Returns:
            list[Message]: A list containing the response from Copilot.

        Raises:
            EmptyResponseException: If the response from Copilot is empty.
            InvalidStatus: If the WebSocket handshake fails with an HTTP status error.
            RuntimeError: If any other error occurs during WebSocket communication.
        """
        self._validate_request(message=message)
        request_piece = message.message_pieces[0]

        logger.info(f"Sending the following prompt to WebSocketCopilotTarget: {request_piece}")

        try:
            prompt_text = request_piece.converted_value
            response_text = await self._connect_and_send(prompt_text)

            if not response_text or not response_text.strip():
                logger.error("Empty response received from Copilot.")
                raise EmptyResponseException(message="Copilot returned an empty response.")
            logger.info(f"Received the following response from WebSocketCopilotTarget: {response_text[:100]}...")

            response_entry = construct_response_from_request(
                request=request_piece, response_text_pieces=[response_text]
            )

            return [response_entry]

        except websockets.exceptions.InvalidStatus as e:
            logger.error(
                f"WebSocket connection failed: {str(e)}\n"
                "Ensure the WEBSOCKET_URL environment variable is correct and valid."
                " For more details about authentication, refer to the class documentation."
            )
            raise

        except Exception as e:
            raise RuntimeError(f"An error occurred during WebSocket communication: {str(e)}") from e
