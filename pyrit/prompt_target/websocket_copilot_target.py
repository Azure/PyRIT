# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import asyncio
import json
import logging
import uuid
from enum import IntEnum
from typing import Optional

import jwt
import websockets

from pyrit.auth import CopilotAuthenticator
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
    Authentication is handled automatically using CopilotAuthenticator, which uses Playwright
    to automate browser login and obtain access tokens.

    Requirements:
        Set the following environment variables:
            - COPILOT_USERNAME: Your Microsoft account username (email).
            - COPILOT_PASSWORD: Your Microsoft account password.

        Install Playwright and its browser dependencies:
            pip install playwright
            playwright install chromium

    Note:
        Only works with licensed Microsoft 365 Copilot. The free Copilot version is not compatible.
        Each target instance creates a new conversation session with unique conversation and session IDs.
    """

    SUPPORTED_DATA_TYPES = {"text"}  # TODO: support more types?

    WEBSOCKET_BASE_URL: str = "wss://substrate.office.com/m365Copilot/Chathub"
    RESPONSE_TIMEOUT_SECONDS: int = 60
    CONNECTION_TIMEOUT_SECONDS: int = 30

    def __init__(
        self,
        *,
        verbose: bool = False,
        max_requests_per_minute: Optional[int] = None,
        model_name: str = "copilot",
        response_timeout_seconds: int = RESPONSE_TIMEOUT_SECONDS,
        authenticator: Optional[CopilotAuthenticator] = None,
    ) -> None:
        """
        Initialize the WebSocketCopilotTarget.

        Args:
            verbose (bool): Enable verbose logging. Defaults to False.
            max_requests_per_minute (Optional[int]): Maximum number of requests per minute.
            model_name (str): The model name. Defaults to "copilot".
            response_timeout_seconds (int): Timeout for receiving responses in seconds. Defaults to 60s.
            authenticator (Optional[CopilotAuthenticator]): Authenticator instance for token management.
                If None, a new CopilotAuthenticator instance will be created with default settings.

        Raises:
            ValueError: If ``response_timeout_seconds`` is not a positive integer.
        """
        if response_timeout_seconds <= 0:
            raise ValueError("response_timeout_seconds must be a positive integer.")

        self._authenticator = authenticator or CopilotAuthenticator()
        self._response_timeout_seconds = response_timeout_seconds

        # These will be generated fresh for each request
        self._session_id: Optional[str] = None
        self._conversation_id: Optional[str] = None

        super().__init__(
            verbose=verbose,
            max_requests_per_minute=max_requests_per_minute,
            endpoint=self.WEBSOCKET_BASE_URL,
            model_name=model_name,
        )

        if self._verbose:
            logger.info("WebSocketCopilotTarget initialized")

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
                msg_type = CopilotMessageType._value2member_map_.get(data.get("type", -1), CopilotMessageType.UNKNOWN)

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
                        logger.debug(f"Full raw message: {message}")
                    results.append((CopilotMessageType.FINAL_CONTENT, bot_text))
                    continue

                results.append((msg_type, ""))

            except json.JSONDecodeError as e:
                logger.error(f"Failed to decode JSON message: {str(e)}")
                results.append((CopilotMessageType.UNKNOWN, ""))

        return results if results else [(CopilotMessageType.UNKNOWN, "")]

    async def _build_websocket_url_async(self) -> str:
        access_token = await self._authenticator.get_token()

        try:
            parsed_token = jwt.decode(access_token, algorithms=["RS256"], options={"verify_signature": False})
        except Exception as e:
            raise ValueError(f"Failed to decode access token: {str(e)}") from e

        tenant_id = parsed_token.get("tid")
        object_id = parsed_token.get("oid")

        if not tenant_id or not object_id:
            raise ValueError(
                "Failed to extract tenant_id (tid) or object_id (oid) from bearer token. "
                f"Token claims: {list(parsed_token.keys())}"
            )

        self._session_id = str(uuid.uuid4())
        self._conversation_id = str(uuid.uuid4())
        client_request_id = str(uuid.uuid4())

        base_url = f"{self.WEBSOCKET_BASE_URL}/{object_id}@{tenant_id}"
        query_params = [
            f"ClientRequestId={client_request_id}",
            f"X-SessionId={self._session_id}",
            f"ConversationId={self._conversation_id}",
            f"access_token={access_token}",
            "X-variants=feature.includeExternal,feature.AssistantConnectorsContentSources,"
            "3S.BizChatWprBoostAssistant,3S.EnableMEFromSkillDiscovery,feature.EnableAuthErrorMessage,"
            "EnableRequestPlugins,feature.EnableSensitivityLabels,feature.IsEntityAnnotationsEnabled,"
            "EnableUnsupportedUrlDetector",
            "source=%22officeweb%22",
            "scenario=OfficeWebIncludedCopilot",
        ]

        websocket_url = f"{base_url}?{'&'.join(query_params)}"
        logger.debug(f"WebSocket URL: {websocket_url}")
        return websocket_url

    def _build_prompt_message(self, prompt: str) -> dict:
        request_id = trace_id = uuid.uuid4().hex

        return {
            "arguments": [
                {
                    "source": "officeweb",
                    "clientCorrelationId": uuid.uuid4().hex,
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
                    "threadLevelGptId": {},
                    "conversationId": self._conversation_id,
                    "traceId": trace_id,
                    "isStartOfSession": True,
                    "productThreadType": "Office",
                    "clientInfo": {"clientPlatform": "web"},
                    "message": {
                        "author": "user",
                        "inputMethod": "Keyboard",
                        "text": prompt,
                        "entityAnnotationTypes": ["People", "File", "Event", "Email", "TeamsMessage"],
                        "requestId": request_id,
                        "locationInfo": {"timeZoneOffset": 0, "timeZone": "UTC"},
                        "locale": "en-US",
                        "messageType": "Chat",
                        "experienceType": "Default",
                    },
                    "plugins": [],
                }
            ],
            "invocationId": "0",
            "target": "chat",
            "type": CopilotMessageType.USER_PROMPT,
        }

    async def _connect_and_send(self, prompt: str) -> str:
        websocket_url = await self._build_websocket_url_async()

        # TODO: explain why PING is not sent here
        inputs = [{"protocol": "json", "version": 1}, self._build_prompt_message(prompt)]
        response = ""

        async with websockets.connect(
            websocket_url,
            open_timeout=self.CONNECTION_TIMEOUT_SECONDS,
            close_timeout=self.CONNECTION_TIMEOUT_SECONDS,
        ) as websocket:
            for input_msg in inputs:
                payload = self._dict_to_websocket(input_msg)
                await websocket.send(payload)

                is_user_input = input_msg.get("type") == CopilotMessageType.USER_PROMPT

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
                                response = content
                            elif msg_type == CopilotMessageType.UNKNOWN:
                                logger.debug("Received unknown or empty message type.")

                        elif msg_type == CopilotMessageType.PING and not is_user_input:
                            stop_polling = True

            return response

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
            logger.info(f"Received the following response from WebSocketCopilotTarget: \n{response_text}")

            response_entry = construct_response_from_request(
                request=request_piece, response_text_pieces=[response_text]
            )

            return [response_entry]

        except websockets.exceptions.InvalidStatus as e:
            logger.error(
                f"WebSocket connection failed: {str(e)}\n"
                "Ensure that COPILOT_USERNAME and COPILOT_PASSWORD environment variables are set correctly."
                " For more details about authentication, refer to the class documentation."
            )
            raise

        except Exception as e:
            raise RuntimeError(f"An error occurred during WebSocket communication: {str(e)}") from e
