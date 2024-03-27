# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import enum
import uuid

from datetime import datetime
from typing import Dict
from uuid import uuid4

from pydantic import BaseModel, ConfigDict
from sqlalchemy import Column, String, DateTime, Float, Enum, JSON, ForeignKey, Index, INTEGER, ARRAY
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.dialects.postgresql import UUID


Base = declarative_base()


class PromptDataType(enum.Enum):
    TEXT = 'text'
    IMAGE = 'image'


class PromptMemoryEntry(Base):  # type: ignore
    """
    Represents the prompt data.

    Attributes:
        __tablename__ (str): The name of the database table.
        __table_args__ (dict): Additional arguments for the database table.
        id (UUID): The unique identifier for the memory entry.
        prompt_entry_type (PromptType): The type of the prompt entry (system, request_segment, response).
        conversation_id (str): The identifier for the conversation which is associated with a single target.
        sequence (int): The order of the conversation within a conversation_id
        timestamp (DateTime): The timestamp of the memory entry.
        labels (Dict[str, str]): The labels associated with the memory entry.
        prompt_metadata (JSON): The metadata associated with the prompt.
        converters (list[PromptConverter]): The converters for the prompt.
        prompt_target (PromptTarget): The target for the prompt.
        original_prompt_data_type (PromptDataType): The data type of the original prompt (text, image)
        original_prompt_text (str): The text of the original prompt. If prompt is an image, it's a link.
        original_prompt_data_sha256 (str): The SHA256 hash of the original prompt data.
        converted_prompt_data_type (PromptDataType): The data type of the converted prompt (text, image)
        converted_prompt_text (str): The text of the converted prompt. If prompt is an image, it's a link.
        converted_prompt_data_sha256 (str): The SHA256 hash of the original prompt data.
        idx_conversation_id (Index): The index for the conversation ID.

    Methods:
        __str__(): Returns a string representation of the memory entry.
    """

    __tablename__ = "PromptMemoryEntries"
    __table_args__ = {"extend_existing": True}
    id = Column(UUID(as_uuid=True), nullable=False, primary_key=True, default=uuid4)
    role: 'Column[ChatMessageRole]' = Column(String, nullable=False)
    conversation_id = Column(String, nullable=False)
    sequence = Column(INTEGER, nullable=False, default=0)
    timestamp = Column(DateTime, nullable=False, default=datetime.utcnow)
    labels: Column[Dict[str, str]] = Column(JSON)
    prompt_metadata = Column(JSON)
    converters: 'Column[list[PromptConverter]]' = Column(JSON)
    prompt_target: 'Column[PromptTarget]' = Column(JSON)

    original_prompt_data_type = Column(Enum(PromptDataType))
    original_prompt_text = Column(String)
    original_prompt_data_sha256 = Column(String)

    converted_prompt_data_type = Column(Enum(PromptDataType))
    converted_prompt_text = Column(String)
    converted_prompt_data_sha256 = Column(String)

    idx_conversation_id = Index("idx_conversation_id", "conversation_id")

    def __str__(self):
        return f"{self.role}: {self.content}"


class EmbeddingData(Base):  # type: ignore
    """
    Represents the embedding data associated with conversation entries in the database.
    Each embedding is linked to a specific conversation entry via a 'uuid'.

    Attributes:
        uuid (UUID): The primary key, which is a foreign key referencing the UUID in the MemoryEntries table.
        embedding (ARRAY(Float)): An array of floats representing the embedding vector.
        embedding_type_name (String): The name or type of the embedding, indicating the model or method used.
    """

    __tablename__ = "EmbeddingData"
    # Allows table redefinition if already defined.
    __table_args__ = {"extend_existing": True}
    id = Column(UUID(as_uuid=True), ForeignKey(f"{PromptMemoryEntry.__tablename__}.id"), primary_key=True)
    embedding = Column(ARRAY(Float))
    embedding_type_name = Column(String)

    def __str__(self):
        return f"{self.id}"


class ConversationMessageWithSimilarity(BaseModel):
    model_config = ConfigDict(extra="forbid")
    role: str
    content: str
    metric: str
    score: float = 0.0


class EmbeddingMessageWithSimilarity(BaseModel):
    model_config = ConfigDict(extra="forbid")
    uuid: uuid.UUID
    metric: str
    score: float = 0.0
