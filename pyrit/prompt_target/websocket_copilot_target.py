# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import asyncio
import json
import logging
import uuid
from enum import IntEnum
from typing import Optional

import websockets
from websockets.exceptions import InvalidStatus

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
    A WebSocket-based prompt target for integrating with Microsoft Copilot.

    This class facilitates communication with Microsoft Copilot over a WebSocket connection.
    Authentication is handled automatically via `CopilotAuthenticator`, which uses Playwright
    to automate browser login and obtain the required access tokens.

    Once authenticated, the target supports multi-turn conversations through server-side
    state management. For each PyRIT conversation, it automatically generates consistent
    `session_id` and `conversation_id` values, enabling Copilot to preserve conversational
    context across multiple turns.

    Because conversation state is managed entirely on the Copilot server, this target does
    not resend conversation history with each request and does not support programmatic
    inspection or manipulation of that history. At present, there appears to be no supported
    mechanism for modifying Copilot's server-side conversation state.

    Note:
        This integration only works with licensed Microsoft 365 Copilot.
        The free version of Copilot is not compatible.

    Important:
        - Ensure the following environment variables are set:
            - ``COPILOT_USERNAME`` - your Microsoft account username (email)
            - ``COPILOT_PASSWORD`` - your Microsoft account password

        - Install `Playwright` and its browser dependencies:
            ``pip install playwright && playwright install chromium``
    """

    SUPPORTED_DATA_TYPES = {"text"}

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
        """
        Convert a dictionary to WebSocket message format.

        SignalR protocol (used by Copilot) requires JSON messages terminated with
        ASCII record separator (\\x1e). Minimal JSON formatting reduces bandwidth.

        https://github.com/dotnet/aspnetcore/blob/main/src/SignalR/docs/specs/HubProtocol.md#json-encoding

        Args:
            data (dict): The data to serialize.

        Returns:
            str: JSON string with record separator appended.
        """
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
        messages = message.split("\x1e")  # record separator

        for message in messages:
            if not message or not message.strip():
                continue

            try:
                data = json.loads(message)
                try:
                    msg_type = CopilotMessageType(data.get("type", -1))
                except ValueError:
                    msg_type = CopilotMessageType.UNKNOWN

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

    async def _build_websocket_url_async(self, *, session_id: str, copilot_conversation_id: str) -> str:
        """
        Build the WebSocket URL with all the required authentication and session parameters.

        Returns:
            str: Complete WebSocket URL with authentication and parameters.

        Raises:
            ValueError: If token cannot be decoded or required claims (tid, oid) are missing.
        """
        access_token = await self._authenticator.get_token()
        token_claims = await self._authenticator.get_claims()

        tenant_id = token_claims.get("tid")
        object_id = token_claims.get("oid")

        if not tenant_id or not object_id:
            raise ValueError(
                "Failed to extract tenant_id (tid) or object_id (oid) from bearer token. "
                f"Token claims: {list(token_claims.keys())}"
            )

        client_request_id = str(uuid.uuid4())

        base_url = f"{self.WEBSOCKET_BASE_URL}/{object_id}@{tenant_id}"
        query_params = [
            f"ClientRequestId={client_request_id}",
            f"X-SessionId={session_id}",
            f"ConversationId={copilot_conversation_id}",
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

    def _build_prompt_message(
        self, *, prompt: str, session_id: str, copilot_conversation_id: str, is_start_of_session: bool
    ) -> dict:
        """
        Construct the prompt message payload for Copilot WebSocket API.

        Builds a comprehensive message structure following Copilot's expected format,
        including session metadata, feature flags, and the user's prompt text.

        Returns:
            dict: The complete message payload ready to be sent via WebSocket.
        """
        request_id = trace_id = uuid.uuid4().hex
        result = {
            "arguments": [
                {
                    "source": "officeweb",
                    "clientCorrelationId": uuid.uuid4().hex,
                    "sessionId": session_id,
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
                    "conversationId": copilot_conversation_id,
                    "traceId": trace_id,
                    "isStartOfSession": is_start_of_session,
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

        logger.debug(f"Built prompt message: {result}")
        return result

    async def _connect_and_send(
        self, *, prompt: str, session_id: str, copilot_conversation_id: str, is_start_of_session: bool
    ) -> str:
        """
        Establish WebSocket connection, send prompt, and await response.

        The method polls for messages, ignoring PARTIAL_RESPONSE streaming updates
        until it receives either FINAL_CONTENT (success), STREAM_END, or UNKNOWN (error).

        Args:
            prompt (str): The user prompt text to send.
            session_id (str): Copilot session identifier.
            copilot_conversation_id (str): Copilot conversation identifier.
            is_start_of_session (bool): Whether this is the first message in the conversation.

        Returns:
            str: The final response text from Copilot.

        Raises:
            TimeoutError: If no response received within the specified timeout period.
            RuntimeError: If WebSocket connection closes unexpectedly, protocol violation occurs,
                or maximum message iterations exceeded.
        """
        websocket_url = await self._build_websocket_url_async(
            session_id=session_id, copilot_conversation_id=copilot_conversation_id
        )

        inputs = [
            {"protocol": "json", "version": 1},  # the handshake message, we expect PING in response
            self._build_prompt_message(  # the actual user prompt, we expect FINAL_CONTENT in response
                prompt=prompt,
                session_id=session_id,
                copilot_conversation_id=copilot_conversation_id,
                is_start_of_session=is_start_of_session,
            ),
        ]
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

                MAX_MESSAGE_ITERATIONS = 1000
                iteration_count = 0
                stop_polling = False

                while not stop_polling:
                    # Prevent infinite loops (e.g. if Copilot somehow never sends a terminating message)
                    iteration_count += 1
                    if iteration_count > MAX_MESSAGE_ITERATIONS:
                        raise RuntimeError(
                            f"Exceeded maximum message iterations ({MAX_MESSAGE_ITERATIONS}) "
                            "while waiting for Copilot response."
                        )

                    try:
                        raw_message = await asyncio.wait_for(
                            websocket.recv(),
                            timeout=self._response_timeout_seconds,
                        )
                    except asyncio.TimeoutError:
                        raise TimeoutError(
                            f"Timed out waiting for Copilot response after {self._response_timeout_seconds} seconds."
                        )

                    if raw_message is None:
                        raise RuntimeError(
                            "WebSocket connection closed unexpectedly: received None from websocket.recv()"
                        )

                    message_str = raw_message if isinstance(raw_message, str) else raw_message.decode("utf-8")
                    parsed_messages = self._parse_raw_message(message_str)

                    for msg_type, content in parsed_messages:
                        if msg_type in (
                            CopilotMessageType.UNKNOWN,
                            CopilotMessageType.FINAL_CONTENT,
                            CopilotMessageType.STREAM_END,
                        ):
                            stop_polling = True
                            # Not breaking here to process all messages in this batch,
                            # possibly including FINAL_CONTENT

                            if msg_type == CopilotMessageType.FINAL_CONTENT:
                                response = content
                            elif msg_type == CopilotMessageType.UNKNOWN:
                                logger.debug("Received unknown or empty message type.")

                        # PING is Copilot's acknowledgment of the protocol handshake (first message in inputs[])
                        # It should arrive after the handshake, not after a user prompt
                        # If we're processing a user prompt and receive PING, something is wrong - ignore it
                        # and keep polling for the actual FINAL_CONTENT response
                        elif msg_type == CopilotMessageType.PING and not is_user_input:
                            stop_polling = True

            return response

    def _validate_request(self, *, message: Message) -> None:
        """
        Validate that the message meets target requirements.

        Args:
            message (Message): The message to validate.

        Raises:
            ValueError: If message contains more than one piece or non-text content.
        """
        n_pieces = len(message.message_pieces)
        if n_pieces != 1:
            raise ValueError(f"This target only supports a single message piece. Received: {n_pieces} pieces.")

        piece_type = message.message_pieces[0].converted_value_data_type
        if piece_type != "text":
            raise ValueError(f"This target only supports text prompt input. Received: {piece_type}.")

    def _is_start_of_session(self, *, conversation_id: str) -> bool:
        """
        Determine if this is the first message in a PyRIT conversation.

        Checks memory for existing conversation history to set the appropriate
        flag for Copilot's server-side conversation initialization.

        Args:
            conversation_id (str): The PyRIT conversation ID.

        Returns:
            bool: True if no prior messages exist in this conversation, False otherwise.
        """
        conversation_history = self._memory.get_conversation(conversation_id=conversation_id)
        return len(conversation_history) == 0

    def _generate_consistent_copilot_ids(self, *, pyrit_conversation_id: str) -> tuple[str, str]:
        """
        Generate consistent Copilot session_id and conversation_id for a PyRIT conversation.

        This uses a deterministic approach to ensure that the same PyRIT conversation_id
        always maps to the same Copilot session identifiers. This enables multi-turn
        conversations while keeping the target stateless.

        Args:
            pyrit_conversation_id (str): The PyRIT conversation ID from the Message.

        Returns:
            tuple[str, str]: A tuple of (session_id, copilot_conversation_id).
        """
        namespace = uuid.UUID("6ba7b810-9dad-11d1-80b4-00c04fd430c8")  # DNS namespace UUID
        session_id = str(uuid.uuid5(namespace, f"session_{pyrit_conversation_id}"))
        copilot_conversation_id = str(uuid.uuid5(namespace, f"copilot_{pyrit_conversation_id}"))

        return session_id, copilot_conversation_id

    @limit_requests_per_minute
    @pyrit_target_retry
    async def send_prompt_async(self, *, message: Message) -> list[Message]:
        """
        Asynchronously send a message to Microsoft Copilot using WebSocket.

        This method enables multi-turn conversations by using consistent session and conversation
        identifiers derived from the PyRIT conversation_id. The Copilot API maintains conversation
        state server-side, so only the current message is sent (no explicit history required).

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

        pyrit_conversation_id = request_piece.conversation_id
        is_start_of_session = self._is_start_of_session(conversation_id=pyrit_conversation_id)

        session_id, copilot_conversation_id = self._generate_consistent_copilot_ids(
            pyrit_conversation_id=pyrit_conversation_id
        )

        logger.info(
            f"Sending prompt to WebSocketCopilotTarget: {request_piece.converted_value} "
            f"(conversation_id={pyrit_conversation_id}, is_start={is_start_of_session})"
        )

        try:
            prompt_text = request_piece.converted_value
            response_text = await self._connect_and_send(
                prompt=prompt_text,
                session_id=session_id,
                copilot_conversation_id=copilot_conversation_id,
                is_start_of_session=is_start_of_session,
            )

            if not response_text or not response_text.strip():
                logger.error("Empty response received from Copilot.")
                raise EmptyResponseException(message="Copilot returned an empty response.")
            logger.info(f"Received response from WebSocketCopilotTarget (length: {len(response_text)} chars)")

            response_entry = construct_response_from_request(
                request=request_piece, response_text_pieces=[response_text]
            )

            return [response_entry]

        except InvalidStatus as e:
            logger.error(
                f"WebSocket connection failed: {str(e)}\n"
                "Ensure that COPILOT_USERNAME and COPILOT_PASSWORD environment variables are set correctly."
                " For more details about authentication, refer to the class documentation."
            )
            raise

        except EmptyResponseException:
            raise

        except Exception as e:
            raise RuntimeError(f"An error occurred during WebSocket communication: {str(e)}") from e
