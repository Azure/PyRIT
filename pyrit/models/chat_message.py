# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import warnings
from typing import Any, Optional, Union

from pydantic import BaseModel, ConfigDict

from pyrit.models.literals import ChatMessageRole

ALLOWED_CHAT_MESSAGE_ROLES = ["system", "user", "assistant", "tool", "developer"]


class ToolCall(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str
    type: str
    function: str


class ChatMessage(BaseModel):
    """
    Represents a chat message for API consumption.

    The content field can be:
    - A simple string for single-part text messages
    - A list of dicts for multipart messages (e.g., text + images)
    """

    model_config = ConfigDict(extra="forbid")
    role: ChatMessageRole
    content: Union[str, list[dict[str, Any]]]
    name: Optional[str] = None
    tool_calls: Optional[list[ToolCall]] = None
    tool_call_id: Optional[str] = None

    def to_json(self) -> str:
        """
        Serialize the ChatMessage to a JSON string.

        Returns:
            A JSON string representation of the message.
        """
        return self.model_dump_json()

    @classmethod
    def from_json(cls, json_str: str) -> "ChatMessage":
        """
        Deserialize a ChatMessage from a JSON string.

        Args:
            json_str: A JSON string representation of a ChatMessage.

        Returns:
            A ChatMessage instance.
        """
        return cls.model_validate_json(json_str)


class ChatMessageListDictContent(ChatMessage):
    """
    Deprecated: Use ChatMessage instead.

    This class exists for backward compatibility and will be removed in a future version.
    """

    def __init__(self, **data: Any) -> None:
        warnings.warn(
            "ChatMessageListDictContent is deprecated and will be removed in 0.13.0. "
            "Use ChatMessage instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(**data)


class ChatMessagesDataset(BaseModel):
    """
    Represents a dataset of chat messages.

    Parameters:
        model_config (ConfigDict): The model configuration.
        name (str): The name of the dataset.
        description (str): The description of the dataset.
        list_of_chat_messages (list[list[ChatMessage]]): A list of chat messages.
    """

    model_config = ConfigDict(extra="forbid")
    name: str
    description: str
    list_of_chat_messages: list[list[ChatMessage]]
