# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import abc
from typing import Generic, List, TypeVar

from pyrit.models import Message

T = TypeVar("T")


class MessageListNormalizer(abc.ABC, Generic[T]):
    """
    Abstract base class for normalizers that return a list of items.

    Subclasses specify the type T (e.g., Message, ChatMessage) that the list contains.
    """

    @abc.abstractmethod
    async def normalize_async(self, messages: List[Message]) -> List[T]:
        """
        Normalize the list of messages into a list of items.

        Args:
            messages: The list of Message objects to normalize.

        Returns:
            A list of normalized items of type T.
        """


class MessageStringNormalizer(abc.ABC):
    """
    Abstract base class for normalizers that return a string representation.

    Use this for formatting messages into text for non-chat targets or context strings.
    """

    @abc.abstractmethod
    async def normalize_string_async(self, messages: List[Message]) -> str:
        """
        Normalize the list of messages into a string representation.

        Args:
            messages: The list of Message objects to normalize.

        Returns:
            A string representation of the messages.
        """
