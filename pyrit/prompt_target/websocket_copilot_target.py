# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import json
import logging
import os
import uuid
from enum import Enum
from typing import Optional

import websockets

from pyrit.exceptions import (
    EmptyResponseException,
    pyrit_target_retry,
)
from pyrit.models import Message, construct_response_from_request
from pyrit.prompt_target import PromptTarget, limit_requests_per_minute

logger = logging.getLogger(__name__)

"""
Useful links:
https://github.com/mbrg/power-pwn/blob/main/src/powerpwn/copilot/copilot_connector/copilot_connector.py
https://labs.zenity.io/p/access-copilot-m365-terminal
"""


class CopilotMessageType(Enum):
    """Enumeration for Copilot WebSocket message types."""

    UNKNOWN = -1
    NEXT_DATA_FRAME = 1  # streaming Copilot responses
    LAST_DATA_FRAME = 2  # the last data frame with final content
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

    # TODO: implement timeouts and retries
    MAX_WAIT_TIME_SECONDS: int = 300
    POLL_INTERVAL_MS: int = 2000

    def __init__(
        self,
        *,
        verbose: bool = False,
        max_requests_per_minute: Optional[int] = None,
        model_name: str = "copilot",
    ) -> None:
        """
        Initialize the WebSocketCopilotTarget.

        Args:
            verbose (bool): Enable verbose logging. Defaults to False.
            max_requests_per_minute (int, Optional): Maximum number of requests per minute.
            model_name (str): The model name. Defaults to "copilot".

        Raises:
            ValueError: If WebSocket URL is not provided, is empty, or has invalid format.
            ValueError: If required parameters are missing or empty in the WebSocket URL.
        """
        self._websocket_url = os.getenv("WEBSOCKET_URL")
        if not self._websocket_url or self._websocket_url.strip() == "":
            raise ValueError("WebSocket URL must be provided through the WEBSOCKET_URL environment variable")

        if not self._websocket_url.startswith(("wss://", "ws://")):
            raise ValueError(
                "WebSocket URL must start with 'wss://' or 'ws://'. "
                f"Received URL starting with: {self._websocket_url[:10]}"
            )

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

        if self._verbose:
            logger.info(f"WebSocketCopilotTarget initialized with conversation_id: {self._conversation_id}")
            logger.info(f"Session ID: {self._session_id}")

    @staticmethod
    def _dict_to_websocket(data: dict) -> str:
        # Produce the smallest possible JSON string, followed by record separator
        return json.dumps(data, separators=(",", ":")) + "\x1e"

    @staticmethod
    def _parse_message(raw_message: str) -> tuple[int, str, dict]:
        """
        Extract actionable content from raw WebSocket frames.

        Args:
            raw_message (str): The raw WebSocket message string.

        Returns:
            tuple: (message_type, content_text, full_data)
        """
        try:
            # https://github.com/dotnet/aspnetcore/blob/main/src/SignalR/docs/specs/HubProtocol.md#json-encoding
            message = message = raw_message.split("\x1e")[0]  # record separator
            if not message:
                return (-1, "", {})

            data = json.loads(message)
            msg_type = data.get("type", -1)

            if msg_type == 6:  # PING
                return (6, "", data)

            if msg_type == 2:  # LAST_DATA_FRAME
                item = data.get("item", {})
                if item:
                    messages = item.get("messages", [])
                    if messages:
                        for msg in reversed(messages):
                            if msg.get("author") == "bot":
                                text = msg.get("text", "")
                                if text:
                                    return (2, text, data)
                # TODO: maybe treat this as error?
                logger.warning("LAST_DATA_FRAME received but no parseable content found.")
                return (2, "", data)

            if msg_type == 1:  # NEXT_DATA_FRAME
                # Streamed updates are not needed for this target
                return (1, "", data)

            return (msg_type, "", data)

        except json.JSONDecodeError as e:
            logger.error(f"Failed to decode JSON message: {str(e)}")
            return (-1, "", {})

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
            "type": 4,
        }

    async def _connect_and_send(self, prompt: str) -> str:
        protocol_msg = {"protocol": "json", "version": 1}
        prompt_dict = self._build_prompt_message(prompt)

        inputs = [protocol_msg, prompt_dict]
        last_response = ""

        async with websockets.connect(self._websocket_url) as websocket:
            for input_msg in inputs:
                payload = self._dict_to_websocket(input_msg)
                is_user_input = input_msg.get("type") == 4  # USER_PROMPT

                await websocket.send(payload)

                stop_polling = False
                while not stop_polling:
                    response = await websocket.recv()
                    msg_type, content, data = self._parse_message(response)

                    if (
                        msg_type in (-1, 2)  # UNKNOWN or LAST_DATA_FRAME
                        or msg_type == 6
                        and not is_user_input
                    ):
                        stop_polling = True

                        if msg_type == 2:  # LAST_DATA_FRAME - final response
                            last_response = content
                        elif msg_type == -1:  # UNKNOWN/NONE
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
