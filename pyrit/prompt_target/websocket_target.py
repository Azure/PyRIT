# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import asyncio
import logging
from collections.abc import Callable
from typing import Any, List, Optional

import websockets

from pyrit.exceptions import (
    pyrit_target_retry,
)
from pyrit.models import Message, construct_response_from_request
from pyrit.prompt_target import PromptTarget, limit_requests_per_minute

logger = logging.getLogger(__name__)


class WebsocketTarget(PromptTarget):
    """
    A general websocket prompt target.

    A list of initialization/connection strings must be provided for establishing a conversation with the target LLM.
    This varies by websocket target and therefore must be provided manually.

    In addition to initialization strings, there is no standard format for websocket messages.
    As such, functions must be provided for both formatting messages to send and parsing responses from the target.

    After establishing a conversation over websocket, the target typically begins the conversation with 1 or more greeting messages.
    The greeting message is discarded so that is not interpreted as a response to the first adversarial prompt.
    The number of greeting messages to discard is dictated by discard_initial_messages argument.
    """

    def __init__(
        self,
        endpoint: str,
        initialization_strings: List[str],
        response_parser: Callable[[str], str],
        message_builder: Callable[[str], str],
        discard_initial_messages: Optional[int] = 1,
        existing_convo: Optional[dict[str, Any]] = None,
        max_requests_per_minute: Optional[int] = None,
        **websockets_kwargs: Any,
    ) -> None:
        """
        Initialize the websocket target with specified parameters.

        Args:
            endpoint (str): the target endpoint
            initialization_strings (List[str]): These are the connection/initialization strings that must be sent after connecting to websocket in order to initiate conversation
            response_parser: (Callable): Function that takes raw websocket message and tries to parse response message; message is discarded if function fails
            message_builder: (Callable): Function that takes prompt and builds the message to send with it
            discard_initial_messages (int): The number of greeting messages that are sent after initialization and should be discarded
            existing_convo (dict[str, websockets.WebSocketClientProtocol], Optional): Existing conversations.
            max_requests_per_minute (int, Optional): Maximum number of requests per minute.
            websockets_kwargs: Additional keyword arguments for websockets connection
        """
        super().__init__(endpoint=endpoint, max_requests_per_minute=max_requests_per_minute)

        self._initialization_strings = initialization_strings
        self._response_parser = response_parser
        self._message_builder = message_builder
        self._discard_initial_messages = discard_initial_messages
        self._existing_conversation = existing_convo if existing_convo is not None else {}
        self._websockets_kwargs = websockets_kwargs or {}

    async def connect(self) -> Any:
        """
        Connect to specified websocket URL.

        Returns:
            The WebSocket connection.
        """
        logger.info(f"Connecting to WebSocket: {self._endpoint}")

        url = self._endpoint

        websocket = await websockets.connect(uri=url, **self._websockets_kwargs)
        logger.info("Successfully connected to websocket")
        return websocket

    async def send_message(self, message: str, conversation_id: str) -> None:
        """
        Send a message to the WebSocket server.

        Args:
            message (str): Message to send in str format.
            conversation_id (str): Conversation ID
        """
        websocket = self._get_websocket(conversation_id=conversation_id)
        await websocket.send(message)
        logger.debug(f"Message sent: {message}")

    @limit_requests_per_minute
    @pyrit_target_retry
    async def send_prompt_async(self, *, message: Message) -> list[Message]:
        """
        Asynchronously send a message to the WebSocket.

        Args:
            message (Message): The message object containing the prompt to send.

        Returns:
            list[Message]: A list containing the response from the prompt target.

        Raises:
            ValueError: If the message piece type is unsupported.
        """
        convo_id = message.message_pieces[0].conversation_id
        if convo_id not in self._existing_conversation:
            websocket = await self.connect()
            self._existing_conversation[convo_id] = websocket

            # Send all necessary connection/initialization strings
            for init_string in self._initialization_strings:
                await self.send_message(message=init_string, conversation_id=convo_id)
                # Give the server a moment to process the session update
                await asyncio.sleep(0.5)

            # Need to make sure bot has finished joining before we proceed
            await asyncio.sleep(5.0)
            # Below loop is to discard greeting message(s)
            for i in range(self._discard_initial_messages):
                result = await self.receive_messages(conversation_id=convo_id)

        websocket = self._existing_conversation[convo_id]

        self._validate_request(message=message)

        request = message.message_pieces[0]

        response_type = request.converted_value_data_type

        if response_type == "text":
            result = await self.send_text_async(text=request.converted_value, conversation_id=convo_id)
        else:
            raise ValueError(f"Unsupported response type: {response_type}")

        text_response_piece = construct_response_from_request(
            request=request, response_text_pieces=[result], response_type="text"
        ).message_pieces[0]

        response_entry = Message(message_pieces=[text_response_piece])
        return [response_entry]

    async def cleanup_target(self) -> None:
        """
        Disconnects from the WebSocket server to clean up, cleaning up all existing conversations.
        """
        for conversation_id, websocket in self._existing_conversation.items():
            if websocket:
                await websocket.close()
                logger.info(f"Disconnected from {self._endpoint} with conversation ID: {conversation_id}")
        self._existing_conversation = {}

    async def cleanup_conversation(self, conversation_id: str) -> None:
        """
        Disconnects from the WebSocket server for a specific conversation.

        Args:
            conversation_id (str): The conversation ID to disconnect from.
        """
        websocket = self._existing_conversation.get(conversation_id)
        if websocket:
            await websocket.close()
            logger.info(f"Disconnected from {self._endpoint} with conversation ID: {conversation_id}")
            del self._existing_conversation[conversation_id]

    async def receive_messages(self, conversation_id: str) -> str:
        """
        Continuously receive messages from the WebSocket server.
        Stops when message is received that contains response (determined from response_parser).

        Args:
            conversation_id: conversation ID

        Returns:
            str: Parsed text from response message

        Raises:
            ConnectionError: If WebSocket connection is not valid
        """
        websocket = self._get_websocket(conversation_id=conversation_id)

        result = ""

        try:
            async for message in websocket:
                try:
                    parsed_message = self._response_parser(message)
                except:
                    parsed_message = None

                if parsed_message:
                    logger.debug(f"Received message: {parsed_message}")
                    result = parsed_message
                    break
                else:
                    logger.debug(f"Websocket message did not contain response from LLM. Continuing.")

        except websockets.ConnectionClosed as e:
            logger.error(f"WebSocket connection closed for conversation {conversation_id}: {e}")
        except Exception as e:
            logger.error(f"An unexpected error occurred for conversation {conversation_id}: {e}")
            raise

        return result

    def _get_websocket(self, *, conversation_id: str) -> Any:
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

    async def send_text_async(self, text: str, conversation_id: str) -> str:
        """
        Send text prompt to the WebSocket server.

        Args:
            text: prompt to send.
            conversation_id: conversation ID

        Returns:
            str: Response from target
        """
        message_w_prompt = self._message_builder(text)

        logger.info(f"Sending text message: {message_w_prompt}")
        await self.send_message(message=message_w_prompt, conversation_id=conversation_id)

        # Listen for responses
        receive_messages = asyncio.create_task(self.receive_messages(conversation_id=conversation_id))

        result = await asyncio.wait_for(receive_messages, timeout=30.0)  # Wait for all responses to be received

        return result

    def _validate_request(self, *, message: Message) -> None:
        """
        Validate the structure and content of a message for compatibility of this target.

        Args:
            message (Message): The message object.

        Raises:
            ValueError: If more than two message pieces are provided.
            ValueError: If any of the message pieces have a data type other than 'text'.
        """
        # Check the number of message pieces
        n_pieces = len(message.message_pieces)
        if n_pieces != 1:
            raise ValueError(f"This target only supports one message piece. Received: {n_pieces} pieces.")

        piece_type = message.message_pieces[0].converted_value_data_type
        if piece_type not in ["text"]:
            raise ValueError(f"This target only supports text prompt input. Received: {piece_type}.")

    def is_json_response_supported(self) -> bool:
        """
        Check if the target supports JSON as a response format.

        Returns:
            bool: True if JSON response is supported, False otherwise.
        """
        return False
