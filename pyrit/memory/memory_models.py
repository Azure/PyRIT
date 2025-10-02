# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import json
import uuid
from datetime import datetime
from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict
from sqlalchemy import (
    ARRAY,
    INTEGER,
    JSON,
    DateTime,
    Float,
    ForeignKey,
    Index,
    String,
    TypeDecorator,
    Unicode,
)
from sqlalchemy.dialects.sqlite import CHAR
from sqlalchemy.orm import (  # type: ignore
    DeclarativeBase,
    Mapped,
    mapped_column,
    relationship,
)
from sqlalchemy.types import Uuid  # type: ignore

from pyrit.common.utils import to_sha256
from pyrit.models import PromptDataType, PromptRequestPiece, Score, SeedPrompt
from pyrit.models.attack_result import AttackOutcome, AttackResult
from pyrit.models.conversation_reference import ConversationReference, ConversationType
from pyrit.models.literals import ChatMessageRole


class CustomUUID(TypeDecorator):
    """
    A custom UUID type that works consistently across different database backends.
    For SQLite, stores UUIDs as strings and converts them back to UUID objects.
    For other databases, uses the native UUID type.
    """

    impl = CHAR
    cache_ok = True

    def load_dialect_impl(self, dialect):
        if dialect.name == "sqlite":
            return dialect.type_descriptor(CHAR(36))
        else:
            return dialect.type_descriptor(Uuid())

    def process_bind_param(self, value, dialect):
        return str(value) if value else None

    def process_result_value(self, value, dialect):
        if dialect.name == "sqlite":
            return uuid.UUID(value) if value else None
        return value


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
        attack_identifier (Dict[str, str]): The attack identifier for the prompt.
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
    id = mapped_column(CustomUUID, nullable=False, primary_key=True)
    role: Mapped[Literal["system", "user", "assistant", "tool", "developer"]] = mapped_column(String, nullable=False)
    conversation_id = mapped_column(String, nullable=False)
    sequence = mapped_column(INTEGER, nullable=False)
    timestamp = mapped_column(DateTime, nullable=False)
    labels: Mapped[dict[str, str]] = mapped_column(JSON)
    prompt_metadata: Mapped[dict[str, Union[str, int]]] = mapped_column(JSON)
    converter_identifiers: Mapped[Optional[List[dict[str, str]]]] = mapped_column(JSON)
    prompt_target_identifier: Mapped[dict[str, str]] = mapped_column(JSON)
    attack_identifier: Mapped[dict[str, str]] = mapped_column(JSON)
    response_error: Mapped[Literal["blocked", "none", "processing", "unknown"]] = mapped_column(String, nullable=True)

    original_value_data_type: Mapped[Literal["text", "image_path", "audio_path", "url", "error"]] = mapped_column(
        String, nullable=False
    )
    original_value = mapped_column(Unicode, nullable=False)
    original_value_sha256 = mapped_column(String)

    converted_value_data_type: Mapped[Literal["text", "image_path", "audio_path", "url", "error"]] = mapped_column(
        String, nullable=False
    )
    converted_value = mapped_column(Unicode)
    converted_value_sha256 = mapped_column(String)

    idx_conversation_id = Index("idx_conversation_id", "conversation_id")

    original_prompt_id = mapped_column(CustomUUID, nullable=False)

    scores: Mapped[List["ScoreEntry"]] = relationship(
        "ScoreEntry",
        primaryjoin="ScoreEntry.prompt_request_response_id == PromptMemoryEntry.original_prompt_id",
        back_populates="prompt_request_piece",
        foreign_keys="ScoreEntry.prompt_request_response_id",
    )

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
        self.attack_identifier = entry.attack_identifier

        self.original_value = entry.original_value
        self.original_value_data_type = entry.original_value_data_type  # type: ignore
        self.original_value_sha256 = entry.original_value_sha256

        self.converted_value = entry.converted_value
        self.converted_value_data_type = entry.converted_value_data_type  # type: ignore
        self.converted_value_sha256 = entry.converted_value_sha256

        self.response_error = entry.response_error  # type: ignore

        self.original_prompt_id = entry.original_prompt_id

    def get_prompt_request_piece(self) -> PromptRequestPiece:
        prompt_request_piece = PromptRequestPiece(
            role=self.role,
            original_value=self.original_value,
            original_value_sha256=self.original_value_sha256,
            converted_value=self.converted_value,
            converted_value_sha256=self.converted_value_sha256,
            id=self.id,
            conversation_id=self.conversation_id,
            sequence=self.sequence,
            labels=self.labels,
            prompt_metadata=self.prompt_metadata,
            converter_identifiers=self.converter_identifiers,
            prompt_target_identifier=self.prompt_target_identifier,
            attack_identifier=self.attack_identifier,
            original_value_data_type=self.original_value_data_type,
            converted_value_data_type=self.converted_value_data_type,
            response_error=self.response_error,
            original_prompt_id=self.original_prompt_id,
            timestamp=self.timestamp,
        )
        prompt_request_piece.scores = [score.get_score() for score in self.scores]
        return prompt_request_piece

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
    id = mapped_column(Uuid(as_uuid=True), ForeignKey(f"{PromptMemoryEntry.__tablename__}.id"), primary_key=True)
    embedding = mapped_column(ARRAY(Float).with_variant(JSON, "sqlite"))  # type: ignore
    embedding_type_name = mapped_column(String)

    def __str__(self):
        return f"{self.id}"


class ScoreEntry(Base):  # type: ignore
    """
    Represents the Score Memory Entry

    """

    __tablename__ = "ScoreEntries"
    __table_args__ = {"extend_existing": True}

    id = mapped_column(CustomUUID, nullable=False, primary_key=True)
    score_value = mapped_column(String, nullable=False)
    score_value_description = mapped_column(String, nullable=True)
    score_type: Mapped[Literal["true_false", "float_scale"]] = mapped_column(String, nullable=False)
    score_category = mapped_column(String, nullable=True)
    score_rationale = mapped_column(String, nullable=True)
    score_metadata = mapped_column(String, nullable=True)
    scorer_class_identifier: Mapped[dict[str, str]] = mapped_column(JSON)
    prompt_request_response_id = mapped_column(CustomUUID, ForeignKey(f"{PromptMemoryEntry.__tablename__}.id"))
    timestamp = mapped_column(DateTime, nullable=False)
    task = mapped_column(String, nullable=True)
    prompt_request_piece: Mapped["PromptMemoryEntry"] = relationship("PromptMemoryEntry", back_populates="scores")

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
    SeedPrompt merely reflects basic prompts before plugging into attacks,
    running through models with corresponding attack strategies, and applying converters.
    PromptMemoryEntry captures the processed prompt data before and after the above steps.

    Parameters:
        __tablename__ (str): The name of the database table.
        __table_args__ (dict): Additional arguments for the database table.
        id (Uuid): The unique identifier for the memory entry.
        value (str): The value of the seed prompt.
        value_sha256 (str): The SHA256 hash of the value of the seed prompt data.
        data_type (PromptDataType): The data type of the seed prompt.
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
        role (str): The role of the prompt (e.g., user, system, assistant).

    Methods:
        __str__(): Returns a string representation of the memory entry.
    """

    __tablename__ = "SeedPromptEntries"
    __table_args__ = {"extend_existing": True}
    id = mapped_column(CustomUUID, nullable=False, primary_key=True)
    value = mapped_column(Unicode, nullable=False)
    value_sha256 = mapped_column(Unicode, nullable=True)
    data_type: Mapped[PromptDataType] = mapped_column(String, nullable=False)
    name = mapped_column(String, nullable=True)
    dataset_name = mapped_column(String, nullable=True)
    harm_categories: Mapped[Optional[List[str]]] = mapped_column(JSON, nullable=True)
    description = mapped_column(String, nullable=True)
    authors: Mapped[Optional[List[str]]] = mapped_column(JSON, nullable=True)
    groups: Mapped[Optional[List[str]]] = mapped_column(JSON, nullable=True)
    source = mapped_column(String, nullable=True)
    date_added = mapped_column(DateTime, nullable=False)
    added_by = mapped_column(String, nullable=False)
    prompt_metadata: Mapped[dict[str, Union[str, int]]] = mapped_column(JSON, nullable=True)
    parameters: Mapped[Optional[List[str]]] = mapped_column(JSON, nullable=True)
    prompt_group_id: Mapped[Optional[uuid.UUID]] = mapped_column(CustomUUID, nullable=True)
    sequence: Mapped[Optional[int]] = mapped_column(INTEGER, nullable=True)
    role: Mapped[ChatMessageRole] = mapped_column(String, nullable=True)

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
        self.role = entry.role

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
            role=self.role,
        )


class AttackResultEntry(Base):
    """
    Represents the attack result data in the database.

    Parameters:
        __tablename__ (str): The name of the database table.
        __table_args__ (dict): Additional arguments for the database table.
        id (Uuid): The unique identifier for the attack result entry.
        conversation_id (str): The unique identifier of the conversation that produced this result.
        objective (str): Natural-language description of the attacker's objective.
        attack_identifier (dict[str, str]): Identifier of the attack (e.g., name, module).
        objective_sha256 (str): The SHA256 hash of the objective.
        last_response_id (Uuid): Foreign key to the last response PromptRequestPiece.
        last_score_id (Uuid): Foreign key to the last score ScoreEntry.
        executed_turns (int): Total number of turns that were executed.
        execution_time_ms (int): Total execution time of the attack in milliseconds.
        outcome (AttackOutcome): The outcome of the attack, indicating success, failure, or undetermined.
        outcome_reason (str): Optional reason for the outcome, providing additional context.
        attack_metadata (dict[str, Any]): Metadata can be included as key-value pairs to provide extra context.
        pruned_conversation_ids (List[str]): List of conversation IDs that were pruned from the attack.
        adversarial_chat_conversation_ids (List[str]): List of conversation IDs used for adversarial chat.
        timestamp (DateTime): The timestamp of the attack result entry.
        last_response (PromptMemoryEntry): Relationship to the last response prompt memory entry.
        last_score (ScoreEntry): Relationship to the last score entry.
    Methods:
        __str__(): Returns a string representation of the attack result entry.
    """

    __tablename__ = "AttackResultEntries"
    __table_args__ = {"extend_existing": True}
    id = mapped_column(CustomUUID, nullable=False, primary_key=True)
    conversation_id = mapped_column(String, nullable=False)
    objective = mapped_column(Unicode, nullable=False)
    attack_identifier: Mapped[dict[str, str]] = mapped_column(JSON, nullable=False)
    objective_sha256 = mapped_column(String, nullable=True)
    last_response_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        CustomUUID, ForeignKey(f"{PromptMemoryEntry.__tablename__}.id"), nullable=True
    )
    last_score_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        CustomUUID, ForeignKey(f"{ScoreEntry.__tablename__}.id"), nullable=True
    )
    executed_turns = mapped_column(INTEGER, nullable=False, default=0)
    execution_time_ms = mapped_column(INTEGER, nullable=False, default=0)
    outcome: Mapped[Literal["success", "failure", "undetermined"]] = mapped_column(
        String, nullable=False, default="undetermined"
    )
    outcome_reason = mapped_column(String, nullable=True)
    attack_metadata: Mapped[dict[str, Union[str, int, float, bool]]] = mapped_column(JSON, nullable=True)
    pruned_conversation_ids: Mapped[Optional[List[str]]] = mapped_column(JSON, nullable=True)
    adversarial_chat_conversation_ids: Mapped[Optional[List[str]]] = mapped_column(JSON, nullable=True)
    timestamp = mapped_column(DateTime, nullable=False)

    last_response: Mapped[Optional["PromptMemoryEntry"]] = relationship(
        "PromptMemoryEntry",
        foreign_keys=[last_response_id],
    )
    last_score: Mapped[Optional["ScoreEntry"]] = relationship(
        "ScoreEntry",
        foreign_keys=[last_score_id],
    )

    def __init__(self, *, entry: AttackResult):
        self.id = uuid.uuid4()
        self.conversation_id = entry.conversation_id
        self.objective = entry.objective
        self.attack_identifier = entry.attack_identifier
        self.objective_sha256 = to_sha256(entry.objective)

        # Use helper method for UUID conversions
        self.last_response_id = self._get_id_as_uuid(entry.last_response)
        self.last_score_id = self._get_id_as_uuid(entry.last_score)

        self.executed_turns = entry.executed_turns
        self.execution_time_ms = entry.execution_time_ms
        self.outcome = entry.outcome.value  # type: ignore
        self.outcome_reason = entry.outcome_reason
        self.attack_metadata = self.filter_json_serializable_metadata(entry.metadata)

        # Persist conversation references by type
        self.pruned_conversation_ids = [
            ref.conversation_id for ref in entry.get_conversations_by_type(ConversationType.PRUNED)
        ] or None

        self.adversarial_chat_conversation_ids = [
            ref.conversation_id for ref in entry.get_conversations_by_type(ConversationType.ADVERSARIAL)
        ] or None

        self.timestamp = datetime.now()

    @staticmethod
    def _get_id_as_uuid(obj: Any) -> Optional[uuid.UUID]:
        """
        Safely extract and convert an object's id to UUID.

        Args:
            obj: Object that might have an id attribute

        Returns:
            UUID if successful, None otherwise
        """
        if obj and hasattr(obj, "id") and obj.id:
            try:
                return uuid.UUID(str(obj.id))
            except (ValueError, TypeError):
                pass
        return None

    @staticmethod
    def filter_json_serializable_metadata(metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Filter a dictionary to only include JSON-serializable values.

        This function iterates through the metadata dictionary and keeps only
        values that can be serialized to JSON, discarding any non-serializable objects.

        Args:
            metadata: Dictionary with potentially non-serializable values

        Returns:
            Dictionary with only JSON-serializable values
        """
        if not metadata:
            return {}

        filtered_metadata = {}

        for key, value in metadata.items():
            try:
                json.dumps(value)
                filtered_metadata[key] = value
            except (TypeError, ValueError):
                pass

        return filtered_metadata

    def get_attack_result(self) -> AttackResult:

        related_conversations: set[ConversationReference] = set()

        for cid in self.pruned_conversation_ids or []:
            related_conversations.add(
                ConversationReference(
                    conversation_id=cid,
                    conversation_type=ConversationType.PRUNED,
                    description="pruned conversation",
                )
            )

        for cid in self.adversarial_chat_conversation_ids or []:
            related_conversations.add(
                ConversationReference(
                    conversation_id=cid,
                    conversation_type=ConversationType.ADVERSARIAL,
                    description="adversarial chat conversation",
                )
            )

        return AttackResult(
            conversation_id=self.conversation_id,
            objective=self.objective,
            attack_identifier=self.attack_identifier,
            last_response=self.last_response.get_prompt_request_piece() if self.last_response else None,
            last_score=self.last_score.get_score() if self.last_score else None,
            executed_turns=self.executed_turns,
            execution_time_ms=self.execution_time_ms,
            outcome=AttackOutcome(self.outcome),
            outcome_reason=self.outcome_reason,
            related_conversations=related_conversations,
            metadata=self.attack_metadata or {},
        )
