# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Any, Optional

from pydantic import BaseModel, ConfigDict

from pyrit.models.literals import ChatMessageRole

ALLOWED_CHAT_MESSAGE_ROLES = ["system", "user", "assistant"]


class ToolCall(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str
    type: str
    function: str


class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="forbid")
    role: ChatMessageRole
    content: str
    name: Optional[str] = None
    tool_calls: Optional[list[ToolCall]] = None
    tool_call_id: Optional[str] = None


class ChatMessageListDictContent(BaseModel):
    model_config = ConfigDict(extra="forbid")
    role: ChatMessageRole
    content: list[dict[str, Any]]  # type: ignore
    name: Optional[str] = None
    tool_calls: Optional[list[ToolCall]] = None
    tool_call_id: Optional[str] = None


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
