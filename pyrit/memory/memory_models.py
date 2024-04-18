# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

# mypy: ignore-errors

import uuid

from pydantic import BaseModel, ConfigDict
from sqlalchemy import Column, String, DateTime, Float, JSON, ForeignKey, Index, INTEGER, ARRAY
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.dialects.postgresql import UUID

from pyrit.models import PromptRequestPiece


Base = declarative_base()


class PromptMemoryEntry(Base):
    """
    Represents the prompt data.

    Because of the nature of database and sql alchemy, type ignores are abundant :)

    Attributes:
        __tablename__ (str): The name of the database table.
        __table_args__ (dict): Additional arguments for the database table.
        id (UUID): The unique identifier for the memory entry.
        role (PromptType): system, assistant, user
        conversation_id (str): The identifier for the conversation which is associated with a single target.
        sequence (int): The order of the conversation within a conversation_id.
            Can be the same number for multi-part requests or multi-part responses.
        timestamp (DateTime): The timestamp of the memory entry.
        labels (Dict[str, str]): The labels associated with the memory entry. Several can be standardized.
        prompt_metadata (JSON): The metadata associated with the prompt. This can be specific to any scenarios.
            Because memory is how components talk with each other, this can be component specific.
            e.g. the URI from a file uploaded to a blob store, or a document type you want to upload.
        converters (list[PromptConverter]): The converters for the prompt.
        prompt_target (PromptTarget): The target for the prompt.
        orchestrator (Orchestrator): The orchestrator for the prompt.
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
    id = Column(UUID(as_uuid=True), nullable=False, primary_key=True)
    role = Column(String, nullable=False)
    conversation_id = Column(String, nullable=False)
    sequence = Column(INTEGER, nullable=False)
    timestamp = Column(DateTime, nullable=False)
    labels = Column(JSON)
    prompt_metadata = Column(String, nullable=True)
    converter_identifiers = Column(JSON)
    prompt_target_identifier = Column(JSON)
    orchestrator_identifier = Column(JSON)
    response_error = Column(String, nullable=True)

    original_prompt_data_type = Column(String, nullable=False)
    original_prompt_text = Column(String, nullable=False)
    original_prompt_data_sha256 = Column(String)

    converted_prompt_data_type = Column(String, nullable=False)
    converted_prompt_text = Column(String)
    converted_prompt_data_sha256 = Column(String)

    idx_conversation_id = Index("idx_conversation_id", "conversation_id")

    def __init__(self, *, entry: PromptRequestPiece):
        self.id = entry.id
        self.role = entry.role
        self.conversation_id = entry.conversation_id
        self.sequence = entry.sequence
        self.timestamp = entry.timestamp
        self.labels = entry.labels
        self.prompt_metadata = entry.prompt_metadata
        self.converter_identifiers = entry.converter_identifiers
        self.prompt_target_identifier = entry.prompt_target_identifier
        self.orchestrator_identifier = entry.orchestrator_identifier

        self.original_prompt_text = entry.original_prompt_text
        self.original_prompt_data_type = entry.original_prompt_data_type
        self.original_prompt_data_sha256 = entry.original_prompt_data_sha256

        self.converted_prompt_data_type = entry.converted_prompt_data_type
        self.converted_prompt_text = entry.converted_prompt_text
        self.converted_prompt_data_sha256 = entry.converted_prompt_data_sha256

        self.response_error = entry.response_error

    def get_prompt_reqest_piece(self) -> PromptRequestPiece:
        return PromptRequestPiece(
            role=self.role,
            original_prompt_text=self.original_prompt_text,
            converted_prompt_text=self.converted_prompt_text,
            id=self.id,
            conversation_id=self.conversation_id,
            sequence=self.sequence,
            labels=self.labels,
            prompt_metadata=self.prompt_metadata,
            converter_identifiers=self.converter_identifiers,
            prompt_target_identifier=self.prompt_target_identifier,
            orchestrator_identifier=self.orchestrator_identifier,
            original_prompt_data_type=self.original_prompt_data_type,
            converted_prompt_data_type=self.converted_prompt_data_type,
            response_error=self.response_error,
        )

    def __str__(self):
        return f"{self.prompt_target_identifier}: {self.role}: {self.converted_prompt_text}"


class EmbeddingData(Base):  # type: ignore
    """
    Represents the embedding data associated with conversation entries in the database.
    Each embedding is linked to a specific conversation entry via an id

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
