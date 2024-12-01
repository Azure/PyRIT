# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import uuid
from typing import List, Literal, Optional

from pydantic import BaseModel, ConfigDict
from pyrit.models import PromptDataType, PromptRequestPiece, Score, SeedPrompt
from sqlalchemy import Column, String, DateTime, Float, JSON, ForeignKey, Index, INTEGER, ARRAY, Unicode
from sqlalchemy.types import Uuid  # type: ignore
from sqlalchemy.orm import DeclarativeBase  # type: ignore
from sqlalchemy.orm import Mapped  # type: ignore


class Base(DeclarativeBase):
    pass


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

    Methods:
        __str__(): Returns a string representation of the memory entry.
    """

    __tablename__ = "PromptMemoryEntries"
    __table_args__ = {"extend_existing": True}
    id = Column(Uuid, nullable=False, primary_key=True)
    role: Mapped[Literal["system", "user", "assistant"]] = Column(String, nullable=False)
    conversation_id = Column(String, nullable=False)
    sequence = Column(INTEGER, nullable=False)
    timestamp = Column(DateTime, nullable=False)
    labels: Mapped[dict[str, str]] = Column(JSON)
    prompt_metadata = Column(String, nullable=True)
    converter_identifiers: Mapped[dict[str, str]] = Column(JSON)
    prompt_target_identifier: Mapped[dict[str, str]] = Column(JSON)
    orchestrator_identifier: Mapped[dict[str, str]] = Column(JSON)
    response_error: Mapped[Literal["blocked", "none", "processing", "unknown"]] = Column(String, nullable=True)

    original_value_data_type: Mapped[Literal["text", "image_path", "audio_path", "url", "error"]] = Column(
        String, nullable=False
    )
    original_value = Column(Unicode, nullable=False)
    original_value_sha256 = Column(String)

    converted_value_data_type: Mapped[Literal["text", "image_path", "audio_path", "url", "error"]] = Column(
        String, nullable=False
    )
    converted_value = Column(Unicode)
    converted_value_sha256 = Column(String)

    idx_conversation_id = Index("idx_conversation_id", "conversation_id")

    original_prompt_id = Column(Uuid, nullable=False)

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

        self.original_value = entry.original_value
        self.original_value_data_type = entry.original_value_data_type
        self.original_value_sha256 = entry.original_value_sha256

        self.converted_value = entry.converted_value
        self.converted_value_data_type = entry.converted_value_data_type
        self.converted_value_sha256 = entry.converted_value_sha256

        self.response_error = entry.response_error

        self.original_prompt_id = entry.original_prompt_id

    def get_prompt_request_piece(self) -> PromptRequestPiece:
        return PromptRequestPiece(
            role=self.role,
            original_value=self.original_value,
            converted_value=self.converted_value,
            id=self.id,
            conversation_id=self.conversation_id,
            sequence=self.sequence,
            labels=self.labels,
            prompt_metadata=self.prompt_metadata,
            converter_identifiers=self.converter_identifiers,
            prompt_target_identifier=self.prompt_target_identifier,
            orchestrator_identifier=self.orchestrator_identifier,
            original_value_data_type=self.original_value_data_type,
            converted_value_data_type=self.converted_value_data_type,
            response_error=self.response_error,
            original_prompt_id=self.original_prompt_id,
            timestamp=self.timestamp,
        )

    def __str__(self):
        if self.prompt_target_identifier:
            return f"{self.prompt_target_identifier['__type__']}: {self.role}: {self.converted_value}"
        return f": {self.role}: {self.converted_value}"


class EmbeddingDataEntry(Base):  # type: ignore
    """
    Represents the embedding data associated with conversation entries in the database.
    Each embedding is linked to a specific conversation entry via an id

    Parameters:
        id (Uuid): The primary key, which is a foreign key referencing the UUID in the PromptMemoryEntries table.
        embedding (ARRAY(Float)): An array of floats representing the embedding vector.
        embedding_type_name (String): The name or type of the embedding, indicating the model or method used.
    """

    __tablename__ = "EmbeddingData"
    # Allows table redefinition if already defined.
    __table_args__ = {"extend_existing": True}
    id = Column(Uuid(as_uuid=True), ForeignKey(f"{PromptMemoryEntry.__tablename__}.id"), primary_key=True)
    embedding = Column(ARRAY(Float).with_variant(JSON, "mssql"))  # type: ignore
    embedding_type_name = Column(String)

    def __str__(self):
        return f"{self.id}"


class ScoreEntry(Base):  # type: ignore
    """
    Represents the Score Memory Entry

    """

    __tablename__ = "ScoreEntries"
    __table_args__ = {"extend_existing": True}

    id = Column(Uuid(as_uuid=True), nullable=False, primary_key=True)
    score_value = Column(String, nullable=False)
    score_value_description = Column(String, nullable=True)
    score_type: Mapped[Literal["true_false", "float_scale"]] = Column(String, nullable=False)
    score_category = Column(String, nullable=False)
    score_rationale = Column(String, nullable=True)
    score_metadata = Column(String, nullable=True)
    scorer_class_identifier: Mapped[dict[str, str]] = Column(JSON)
    prompt_request_response_id = Column(Uuid(as_uuid=True), ForeignKey(f"{PromptMemoryEntry.__tablename__}.id"))
    timestamp = Column(DateTime, nullable=False)
    task = Column(String, nullable=True)

    def __init__(self, *, entry: Score):
        self.id = entry.id
        self.score_value = entry.score_value
        self.score_value_description = entry.score_value_description
        self.score_type = entry.score_type
        self.score_category = entry.score_category
        self.score_rationale = entry.score_rationale
        self.score_metadata = entry.score_metadata
        self.scorer_class_identifier = entry.scorer_class_identifier
        self.prompt_request_response_id = entry.prompt_request_response_id if entry.prompt_request_response_id else None
        self.timestamp = entry.timestamp
        self.task = entry.task

    def get_score(self) -> Score:
        return Score(
            id=self.id,
            score_value=self.score_value,
            score_value_description=self.score_value_description,
            score_type=self.score_type,
            score_category=self.score_category,
            score_rationale=self.score_rationale,
            score_metadata=self.score_metadata,
            scorer_class_identifier=self.scorer_class_identifier,
            prompt_request_response_id=self.prompt_request_response_id,
            timestamp=self.timestamp,
            task=self.task,
        )


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
    SeedPrompt merely reflects basic prompts before plugging into orchestrators,
    running through models with corresponding attack strategies, and applying converters.
    PromptMemoryEntry captures the processed prompt data before and after the above steps.

    Parameters:
        __tablename__ (str): The name of the database table.
        __table_args__ (dict): Additional arguments for the database table.
        id (Uuid): The unique identifier for the memory entry.
        value (str): The value of the seed prompt.
        data_type (PromptDataType): The data type of the seed prompt.
        dataset_name (str): The name of the dataset the seed prompt belongs to.
        harm_categories (List[str]): The harm categories associated with the seed prompt.
        description (str): The description of the seed prompt.
        authors (List[str]): The authors of the seed prompt.
        groups (List[str]): The groups involved in authoring the seed prompt (if any).
        source (str): The source of the seed prompt.
        date_added (DateTime): The date the seed prompt was added.
        added_by (str): The user who added the seed prompt.
        prompt_metadata (dict[str, str]): The metadata associated with the seed prompt.
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
    id = Column(Uuid, nullable=False, primary_key=True)
    value = Column(Unicode, nullable=False)
    data_type: Mapped[PromptDataType] = Column(String, nullable=False)
    name = Column(String, nullable=True)
    dataset_name = Column(String, nullable=True)
    harm_categories: Mapped[Optional[List[str]]] = Column(JSON, nullable=True)
    description = Column(String, nullable=True)
    authors: Mapped[Optional[List[str]]] = Column(JSON, nullable=True)
    groups: Mapped[Optional[List[str]]] = Column(JSON, nullable=True)
    source = Column(String, nullable=True)
    date_added = Column(DateTime, nullable=False)
    added_by = Column(String, nullable=False)
    prompt_metadata: Mapped[dict[str, str]] = Column(JSON, nullable=True)
    parameters: Mapped[Optional[List[str]]] = Column(JSON, nullable=True)
    prompt_group_id: Mapped[Optional[uuid.UUID]] = Column(Uuid, nullable=True)
    sequence: Mapped[Optional[int]] = Column(INTEGER, nullable=True)

    def __init__(self, *, entry: SeedPrompt):
        self.id = entry.id
        self.value = entry.value
        self.data_type = entry.data_type
        self.name = entry.name
        self.dataset_name = entry.dataset_name
        self.harm_categories = entry.harm_categories
        self.description = entry.description
        self.authors = entry.authors
        self.groups = entry.groups
        self.source = entry.source
        self.date_added = entry.date_added
        self.added_by = entry.added_by
        self.prompt_metadata = entry.metadata
        self.parameters = entry.parameters
        self.prompt_group_id = entry.prompt_group_id
        self.sequence = entry.sequence

    def get_seed_prompt(self) -> SeedPrompt:
        return SeedPrompt(
            id=self.id,
            value=self.value,
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
