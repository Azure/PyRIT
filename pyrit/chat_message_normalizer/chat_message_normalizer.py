# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import abc
from typing import Generic, TypeVar

from pyrit.models import ChatMessage

T = TypeVar("T", str, list[ChatMessage])


class ChatMessageNormalizer(abc.ABC, Generic[T]):
    @abc.abstractmethod
    def normalize(self, messages: list[ChatMessage]) -> T:
        """
        Normalizes the list of chat messages into a compatible format for the model or target
        """

    @staticmethod
    def squash_system_message(messages: list[ChatMessage], squash_function) -> list[ChatMessage]:
        """
        Combines the system message into the first user request.

        Args:
            messages: The list of chat messages.
            squash_function: The function to combine the system message with the user message.

        Returns:
            The list of chat messages with squashed system messages.
        """
        if not messages:
            raise ValueError("ChatMessage list cannot be empty")

        if messages[0].role == "system":
            if len(messages) == 1:
                return [ChatMessage(role="user", content=messages[0].content)]

            first_user_message = squash_function(messages[0], messages[1])
            return [first_user_message] + messages[2:]

        return messages
