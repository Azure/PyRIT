# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from uuid import uuid4
import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict
from sqlalchemy.orm import declarative_base
from sqlalchemy import Column, String, DateTime, Float
from sqlalchemy.dialects.postgresql import ARRAY, UUID
from sqlalchemy import ForeignKey, Index


Base = declarative_base()


class ConversationData(Base):
    """
    Represents the conversation data.

    conversation_id is used to group messages together within a prompt_target endpoint.
    It's often needed so the prompt_target knows how to construct the messages.

    normalizer_id is used to group messages together within a prompt_normalizer.
    A prompt_normalizer is usually a single attack, and can contain multiple prompt_targets.
    It's often needed to group all the prompts in an attack together.

    Attributes:
        uuid (UUID): A unique identifier for each conversation entry, serving as the primary key.
        role (String): The role associated with the message, indicating its origin 
        within the conversation (e.g., "user", "assistant" or "system").
        content (String): The actual text content of the conversation entry.
        conversation_id (String): An identifier used to group related conversation entries. 
        The conversation_id is linked to a specific LLM model,
        aggregating all related conversations under a single identifier. 
        In scenarios involving multi-turn interactions that utilize two models,
        there will be two distinct conversation_ids, one for each model.
        timestamp (DateTime): The timestamp when the conversation entry was created or 
        logged. Defaults to the current UTC time.
        normalizer_id (String): An identifier used to group messages together within a prompt_normalizer.
        sha256 (String): An optional SHA-256 hash of the content.
        labels (ARRAY(String)): An array of labels associated with the conversation entry, 
        useful for categorization or filtering the final data.
        idx_conversation_id (Index): An index on the `conversation_id` column to improve 
        query performance for operations involving obtaining conversation history based 
        on conversation_id.
    """

    __tablename__ = "ConversationStore"
    __table_args__ = {"extend_existing": True}
    uuid = Column(UUID(as_uuid=True), nullable=False, primary_key=True, default=uuid4)
    role = Column(String, nullable=False)
    content = Column(String)
    conversation_id = Column(String, nullable=False)
    timestamp = Column(DateTime, nullable=False, default=datetime.utcnow)
    normalizer_id = Column(String)
    sha256 = Column(String)
    labels = Column(ARRAY(String))
    idx_conversation_id = Index("idx_conversation_id", "conversation_id")

    def __str__(self):
        return f"{self.role}: {self.content}"


class EmbeddingData(Base):
    """
    Represents the embedding data associated with conversation entries in the database.
    Each embedding is linked to a specific conversation entry via a 'uuid'.

    Attributes:
        uuid (UUID): The primary key, which is a foreign key referencing the UUID in the ConversationStore table.
        embedding (ARRAY(Float)): An array of floats representing the embedding vector.
        embedding_type_name (String): The name or type of the embedding, indicating the model or method used.
    """

    __tablename__ = "EmbeddingStore"
    # Allows table redefinition if already defined.
    __table_args__ = {"extend_existing": True}
    uuid = Column(UUID(as_uuid=True), ForeignKey(f"{ConversationData.__tablename__}.uuid"), primary_key=True)
    embedding = Column(ARRAY(Float))
    embedding_type_name = Column(String)

    def __str__(self):
        return f"{self.uuid}"


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
