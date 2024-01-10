# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import time
from typing import Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, Extra, Field


class EmbeddingMemoryData(BaseModel, extra=Extra.forbid):
    uuid: UUID = Field(default_factory=uuid4)
    embedding: list[float]
    embedding_type_name: str


class ConversationMemoryEntry(BaseModel, extra=Extra.forbid):
    role: str
    content: str
    session: str
    timestamp_in_ns: int = Field(default_factory=time.time_ns)
    uuid: UUID = Field(default_factory=uuid4)
    sha256: str = ""
    embedding_memory_data: Optional[EmbeddingMemoryData] = None
    labels: Optional[list[str]] = None


# This class is convenient for serialization
class ConversationMemoryEntryList(BaseModel, extra=Extra.forbid):
    conversations: list[ConversationMemoryEntry]


class ConversationMessageWithSimilarity(BaseModel, extra=Extra.forbid):
    role: str
    content: str
    metric: str
    score: float = 0.0
