# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import abc
from typing import Any, Generic, List, Protocol, TypeVar

from pyrit.models import Message


class DictConvertible(Protocol):
    """Protocol for objects that can be converted to a dictionary."""

    def to_dict(self) -> dict[str, Any]:
        """Convert the object to a dictionary representation."""
        ...


T = TypeVar("T", bound=DictConvertible)


class MessageListNormalizer(abc.ABC, Generic[T]):
    """
    Abstract base class for normalizers that return a list of items.

    Subclasses specify the type T (e.g., Message, ChatMessage) that the list contains.
    T must implement the DictConvertible protocol (have a to_dict() method).
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

    async def normalize_to_dicts_async(self, messages: List[Message]) -> List[dict[str, Any]]:
        """
        Normalize the list of messages into a list of dictionaries.

        This method uses normalize_async and calls to_dict() on each item.

        Args:
            messages: The list of Message objects to normalize.

        Returns:
            A list of dictionaries representing the normalized messages.
        """
        normalized = await self.normalize_async(messages)
        return [item.to_dict() for item in normalized]


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
