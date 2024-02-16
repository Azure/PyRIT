# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import time
from typing import Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field

from pyrit.models import ChatMessageRole


class EmbeddingMemoryData(BaseModel):
    model_config = ConfigDict(extra="forbid")
    uuid: UUID = Field(default_factory=uuid4)
    embedding: list[float]
    embedding_type_name: str


class ConversationMemoryEntry(BaseModel):
    """
    Represents a single memory entry.

    conversation_id is used to group messages together within a prompt_target endpoint.
    It's often needed so the prompt_target knows how to construct the messages.

    normalizer_id is used to group messages together within a prompt_normalizer.
    A prompt_normalizer is usually a single attack, and can contain multiple prompt_targets.
    It's often needed to group all the prompts in an attack together.

    A memory_entry can contain references to other tables, such as embedding_memory_data, or
    future references like scoring information.
    """

    model_config = ConfigDict(extra="forbid")
    role: ChatMessageRole
    content: str
    conversation_id: str
    timestamp_in_ns: int = Field(default_factory=time.time_ns)
    uuid: UUID = Field(default_factory=uuid4)
    normalizer_id: Optional[str] = None
    sha256: str = ""
    embedding_memory_data: Optional[EmbeddingMemoryData] = None
    labels: Optional[list[str]] = None

    def __str__(self):
        return f"{self.role}: {self.content}"


# This class is convenient for serialization
class ConversationMemoryEntryList(BaseModel):
    model_config = ConfigDict(extra="forbid")
    conversations: list[ConversationMemoryEntry]


class ConversationMessageWithSimilarity(BaseModel):
    model_config = ConfigDict(extra="forbid")
    role: str
    content: str
    metric: str
    score: float = 0.0
