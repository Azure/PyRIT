# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import uuid
from typing import Dict, List, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict
from sqlalchemy import (
    ARRAY,
    INTEGER,
    JSON,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Index,
    String,
    Unicode,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy.types import TypeDecorator

from pyrit.models import PromptDataType, PromptRequestPiece, Score, ScoreType, SeedPrompt

# Create the base class for SQLAlchemy 1.4
Base = declarative_base()

# Custom UUID type for SQLAlchemy 1.4
class GUID(TypeDecorator):
    """Platform-independent GUID type.
    Uses PostgreSQL's UUID type, otherwise uses
    String(36), storing as stringified hex values.
    """
    impl = String(36)
    cache_ok = True

    def load_dialect_impl(self, dialect):
        if dialect.name == 'postgresql':
            return dialect.type_descriptor(UUID())
        else:
            return dialect.type_descriptor(String(36))

    def process_bind_param(self, value, dialect):
        if value is None:
            return value
        elif dialect.name == 'postgresql':
            return str(value)
        else:
            if not isinstance(value, uuid.UUID):
                return str(uuid.UUID(value))
            else:
                return str(value)

    def process_result_value(self, value, dialect):
        if value is None:
            return value
        else:
            if not isinstance(value, uuid.UUID):
                value = uuid.UUID(value)
            return value


class PromptMemoryEntry(Base):
    """
    Represents the prompt data.

    Because of the nature of database and sql alchemy, type ignores are abundant :)

    Parameters:
        __tablename__ (str): The name of the database table.
        __table_args__ (dict): Additional arguments for the database table.
        id (Uuid): The unique identifier for the memory entry.
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
        orchestrator_identifier (Dict[str, str]): The orchestrator identifier for the prompt.
        original_value_data_type (PromptDataType): The data type of the original prompt (text, image)
        original_value (str): The text of the original prompt. If prompt is an image, it's a link.
        original_value_sha256 (str): The SHA256 hash of the original prompt data.
        converted_value_data_type (PromptDataType): The data type of the converted prompt (text, image)
        converted_value (str): The text of the converted prompt. If prompt is an image, it's a link.
        converted_value_sha256 (str): The SHA256 hash of the original prompt data.
        idx_conversation_id (Index): The index for the conversation ID.
        original_prompt_id (UUID): The original prompt id. It is equal to id unless it is a duplicate.
        scores (list[ScoreEntry]): The list of scores associated with the prompt.
    Methods:
        __str__(): Returns a string representation of the memory entry.
    """

    __tablename__ = "PromptMemoryEntries"
    __table_args__ = {"extend_existing": True}
    id = Column(GUID, nullable=False, primary_key=True)
    role = Column(String, nullable=False)
    conversation_id = Column(String, nullable=False)
    sequence = Column(INTEGER, nullable=False)
    timestamp = Column(DateTime, nullable=False)
    labels = Column(JSON)
    prompt_metadata = Column(JSON)
    converter_identifiers = Column(JSON)
    prompt_target_identifier = Column(JSON)
    orchestrator_identifier = Column(JSON)
    response_error = Column(String, nullable=True)

    original_value_data_type = Column(
        String, nullable=False
    )
    original_value = Column(Unicode, nullable=False)
    original_value_sha256 = Column(String, nullable=True)
    converted_value_data_type = Column(
        String, nullable=True
    )
    converted_value = Column(Unicode, nullable=True)
    converted_value_sha256 = Column(String, nullable=True)
    original_prompt_id = Column(GUID, nullable=True)

    # Create an index on conversation_id for faster lookups
    idx_conversation_id = Index("idx_conversation_id", conversation_id)

    # Relationship with ScoreEntry will be defined after ScoreEntry class is defined
    # scores = relationship("ScoreEntry", back_populates="prompt_memory_entry")

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
        self.response_error = entry.response_error

        self.original_value_data_type = entry.original_value_data_type
        self.original_value = entry.original_value
        self.original_value_sha256 = entry.original_value_sha256

        self.converted_value = entry.converted_value
        self.converted_value_data_type = entry.converted_value_data_type
        self.converted_value_sha256 = entry.converted_value_sha256
        self.original_prompt_id = entry.original_prompt_id

    def get_prompt_request_piece(self) -> PromptRequestPiece:
        return PromptRequestPiece(
            id=self.id,
            role=self.role,
            conversation_id=self.conversation_id,
            sequence=self.sequence,
            timestamp=self.timestamp,
            labels=self.labels,
            prompt_metadata=self.prompt_metadata,
            converter_identifiers=self.converter_identifiers,
            prompt_target_identifier=self.prompt_target_identifier,
            orchestrator_identifier=self.orchestrator_identifier,
            response_error=self.response_error,
            original_value_data_type=self.original_value_data_type,
            original_value=self.original_value,
            original_value_sha256=self.original_value_sha256,
            converted_value_data_type=self.converted_value_data_type,
            converted_value=self.converted_value,
            converted_value_sha256=self.converted_value_sha256,
            original_prompt_id=self.original_prompt_id,
        )

    def __str__(self):
        return f"PromptMemoryEntry(id={self.id}, role={self.role}, conversation_id={self.conversation_id})"


class EmbeddingDataEntry(Base):
    """Represents the embedding data associated with conversation entries in the database.
    Each embedding is linked to a specific conversation entry via an id

    Parameters:
        __tablename__ (str): The name of the database table.
        __table_args__ (dict): Additional arguments for the database table.
        id (UUID): The unique identifier for the embedding data entry.
        embedding (List[float]): The embedding vector.
        embedding_type_name (str): The name of the embedding type.
    """

    __tablename__ = "EmbeddingDataEntries"
    __table_args__ = {"extend_existing": True}
    id = Column(GUID, ForeignKey(f"{PromptMemoryEntry.__tablename__}.id"), primary_key=True)
    embedding = Column(ARRAY(Float).with_variant(JSON, "mssql"))
    embedding_type_name = Column(String)

    def __str__(self):
        return f"EmbeddingDataEntry(id={self.id}, embedding_type_name={self.embedding_type_name})"


class ScoreEntry(Base):  # type: ignore
    """
    Represents the Score Memory Entry
    """

    __tablename__ = "ScoreEntries"
    __table_args__ = {"extend_existing": True}

    id = Column(GUID, nullable=False, primary_key=True)
    score_value = Column(String, nullable=False)
    score_value_description = Column(String, nullable=True)
    score_type = Column(String, nullable=False)  # Will store ScoreType values
    score_category = Column(String, nullable=False)
    score_rationale = Column(String, nullable=True)
    score_metadata = Column(String, nullable=True)
    scorer_class_identifier = Column(JSON)  # Dict[str, str]
    prompt_request_response_id = Column(GUID, ForeignKey(f"{PromptMemoryEntry.__tablename__}.id"))
    timestamp = Column(DateTime, nullable=False)
    task = Column(String, nullable=True)
    prompt_request_piece = relationship("PromptMemoryEntry", back_populates="scores")

    def __init__(self, *, entry: Score):
        self.id = entry.id
        self.score_value = entry.score_value
        self.score_value_description = entry.score_value_description
        # Ensure score_type is a valid ScoreType value
        if entry.score_type in ("true_false", "float_scale"):
            self.score_type = entry.score_type
        else:
            raise ValueError(f"Invalid score_type: {entry.score_type}")
        self.score_category = entry.score_category
        self.score_rationale = entry.score_rationale
        self.score_metadata = entry.score_metadata
        self.scorer_class_identifier = entry.scorer_class_identifier
        # Handle None case for prompt_request_response_id
        self.prompt_request_response_id = entry.prompt_request_response_id if entry.prompt_request_response_id else None
        self.timestamp = entry.timestamp
        self.task = entry.task

    def get_score(self) -> Score:
        # Cast score_type to ScoreType since we validated it in __init__
        score_type_val = self.score_type
        if score_type_val not in ("true_false", "float_scale"):
            raise ValueError(f"Invalid score_type: {score_type_val}")
            
        return Score(
            id=self.id,
            score_value=self.score_value,
            score_value_description=self.score_value_description,
            score_type=score_type_val,  # type: ignore
            score_category=self.score_category,
            score_rationale=self.score_rationale,
            score_metadata=self.score_metadata,
            scorer_class_identifier=self.scorer_class_identifier,
            prompt_request_response_id=self.prompt_request_response_id if self.prompt_request_response_id else None,
            timestamp=self.timestamp,
            task=self.task,
        )

    def to_dict(self) -> dict:
        return {
            "id": str(self.id),
            "score_value": self.score_value,
            "score_value_description": self.score_value_description,
            "score_type": self.score_type,
            "score_category": self.score_category,
            "score_rationale": self.score_rationale,
            "score_metadata": self.score_metadata,
            "scorer_class_identifier": self.scorer_class_identifier,
            "prompt_request_response_id": str(self.prompt_request_response_id),
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "task": self.task,
        }


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


class SeedPromptEntry(Base):
    """
    Represents the raw prompt or prompt template data as found in open datasets.

    Note: This is different from the PromptMemoryEntry which is the processed prompt data.

    Parameters:
        __tablename__ (str): The name of the database table.
        __table_args__ (dict): Additional arguments for the database table.
        id (Uuid): The unique identifier for the seed prompt.
        value (str): The text of the seed prompt.
        value_sha256 (str): The SHA256 hash of the seed prompt.
        data_type (PromptDataType): The data type of the seed prompt (text, image).
        name (str): The name of the seed prompt.
        dataset_name (str): The name of the dataset the seed prompt belongs to.
        harm_categories (List[str]): The harm categories associated with the seed prompt.
        description (str): The description of the seed prompt.
        authors (List[str]): The authors of the seed prompt.
        groups (List[str]): The groups involved in authoring the seed prompt (if any).
        source (str): The source of the seed prompt.
        date_added (DateTime): The date the seed prompt was added.
        added_by (str): The user who added the seed prompt.
        prompt_metadata (dict[str, str | int]): The metadata associated with the seed prompt. This includes
            information that is useful for the specific target you're probing, such as encoding data.
        parameters (List[str]): The parameters included in the value.
            Note that seed prompts do not have parameters, only prompt templates do.
            However, they are stored in the same table.
        prompt_group_id (uuid.UUID): The ID of a group the seed prompt may optionally belong to.
            Groups are used to organize prompts for multi-turn conversations or multi-modal prompts.
        sequence (int): The turn of the seed prompt in a group. When entire multi-turn conversations
            are stored, this is used to order the prompts.

    Methods:
        __str__(): Returns a string representation of the memory entry.
    """

    __tablename__ = "SeedPromptEntries"
    __table_args__ = {"extend_existing": True}
    id = Column(GUID, nullable=False, primary_key=True)
    value = Column(Unicode, nullable=False)
    value_sha256 = Column(Unicode, nullable=True)
    data_type = Column(String, nullable=False)
    name = Column(String, nullable=True)
    dataset_name = Column(String, nullable=True)
    harm_categories = Column(JSON, nullable=True)
    description = Column(String, nullable=True)
    authors = Column(JSON, nullable=True)
    groups = Column(JSON, nullable=True)
    source = Column(String, nullable=True)
    date_added = Column(DateTime, nullable=False)
    added_by = Column(String, nullable=False)
    prompt_metadata = Column(JSON, nullable=True)
    parameters = Column(JSON, nullable=True)
    prompt_group_id = Column(GUID, nullable=True)
    sequence = Column(INTEGER, nullable=True)

    def __init__(self, *, entry: SeedPrompt):
        self.id = entry.id
        self.value = entry.value
        self.value_sha256 = entry.value_sha256
        self.data_type = entry.data_type
        self.name = entry.name
        self.dataset_name = entry.dataset_name
        self.harm_categories = entry.harm_categories  # type: ignore
        self.description = entry.description
        self.authors = entry.authors  # type: ignore
        self.groups = entry.groups  # type: ignore
        self.source = entry.source
        self.date_added = entry.date_added
        self.added_by = entry.added_by
        self.prompt_metadata = entry.metadata
        self.parameters = entry.parameters  # type: ignore
        self.prompt_group_id = entry.prompt_group_id
        self.sequence = entry.sequence

    def get_seed_prompt(self) -> SeedPrompt:
        return SeedPrompt(
            id=self.id,
            value=self.value,
            value_sha256=self.value_sha256,
            data_type=self.data_type,
            name=self.name,
            dataset_name=self.dataset_name,
            harm_categories=self.harm_categories,
            description=self.description,
            authors=self.authors,
            groups=self.groups,
            source=self.source,
            date_added=self.date_added,
            added_by=self.added_by,
            metadata=self.prompt_metadata,
            parameters=self.parameters,
            prompt_group_id=self.prompt_group_id,
            sequence=self.sequence,
        )

# Now set up the relationships that couldn't be defined earlier due to forward references

# Add relationship between PromptMemoryEntry and ScoreEntry
PromptMemoryEntry.scores = relationship("ScoreEntry", backref="prompt_memory_entry")
