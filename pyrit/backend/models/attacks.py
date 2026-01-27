# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
Attack-related request and response models.

All interactions in the UI are modeled as "attacks" - including manual conversations.
This is the attack-centric API design where every user interaction targets a model.
"""

from datetime import datetime
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field

from pyrit.backend.models.common import PaginationInfo
from pyrit.backend.models.converters import InlineConverterConfig


class Score(BaseModel):
    """A score associated with a message piece."""

    score_id: str = Field(..., description="Unique score identifier")
    scorer_type: str = Field(..., description="Type of scorer (e.g., 'bias', 'toxicity')")
    score_value: float = Field(..., description="Numeric score value")
    score_rationale: Optional[str] = Field(None, description="Explanation for the score")
    scored_at: datetime = Field(..., description="When the score was generated")


class MessagePiece(BaseModel):
    """
    A piece of a message (text, image, audio, etc.).

    Supports multimodal content with original/converted values and embedded scores.
    Media content is base64-encoded since frontend can't access server file paths.
    """

    piece_id: str = Field(..., description="Unique piece identifier")
    data_type: str = Field(..., description="Data type: 'text', 'image', 'audio', 'video', etc.")
    original_value: Optional[str] = Field(None, description="Original value before conversion")
    original_value_mime_type: Optional[str] = Field(None, description="MIME type of original value")
    converted_value: str = Field(..., description="Converted value (text or base64 for media)")
    converted_value_mime_type: Optional[str] = Field(None, description="MIME type of converted value")
    scores: List[Score] = Field(default_factory=list, description="Scores embedded in this piece")


class Message(BaseModel):
    """A message within an attack."""

    message_id: str = Field(..., description="Unique message identifier")
    turn_number: int = Field(..., description="Turn number in the conversation (1-indexed)")
    role: Literal["user", "assistant", "system"] = Field(..., description="Message role")
    pieces: List[MessagePiece] = Field(..., description="Message pieces (multimodal support)")
    created_at: datetime = Field(..., description="Message creation timestamp")


# ============================================================================
# Attack Summary (List View)
# ============================================================================


class AttackSummary(BaseModel):
    """Summary view of an attack (for list views, omits full message content)."""

    attack_id: str = Field(..., description="Unique attack identifier")
    name: Optional[str] = Field(None, description="Attack name/label")
    target_id: str = Field(..., description="Target instance ID")
    target_type: str = Field(..., description="Target type (e.g., 'azure_openai')")
    outcome: Optional[Literal["pending", "success", "failure"]] = Field(
        None, description="Attack outcome (null if not yet determined)"
    )
    last_message_preview: Optional[str] = Field(
        None, description="Preview of the last message (truncated to ~100 chars)"
    )
    message_count: int = Field(0, description="Total number of messages in the attack")
    created_at: datetime = Field(..., description="Attack creation timestamp")
    updated_at: datetime = Field(..., description="Last update timestamp")


# ============================================================================
# Attack Detail (Single Attack View)
# ============================================================================


class AttackDetail(BaseModel):
    """Detailed view of an attack (includes all messages)."""

    attack_id: str = Field(..., description="Unique attack identifier")
    name: Optional[str] = Field(None, description="Attack name/label")
    target_id: str = Field(..., description="Target instance ID")
    target_type: str = Field(..., description="Target type (e.g., 'azure_openai')")
    outcome: Optional[Literal["pending", "success", "failure"]] = Field(
        None, description="Attack outcome"
    )
    prepended_conversation: List[Message] = Field(
        default_factory=list, description="Prepended messages (system prompts, branching context)"
    )
    messages: List[Message] = Field(default_factory=list, description="Attack messages in order")
    converter_ids: List[str] = Field(default_factory=list, description="Converter instance IDs applied")
    created_at: datetime = Field(..., description="Attack creation timestamp")
    updated_at: datetime = Field(..., description="Last update timestamp")


# ============================================================================
# Attack List Response (Paginated)
# ============================================================================


class AttackListResponse(BaseModel):
    """Paginated response for listing attacks."""

    items: List[AttackSummary] = Field(..., description="List of attack summaries")
    pagination: PaginationInfo = Field(..., description="Pagination metadata")


# ============================================================================
# Create Attack
# ============================================================================


class PrependedMessageRequest(BaseModel):
    """A message to prepend to the attack (for system prompt/branching)."""

    role: Literal["user", "assistant", "system"] = Field(..., description="Message role")
    content: str = Field(..., description="Message content (text)")


class CreateAttackRequest(BaseModel):
    """Request to create a new attack."""

    name: Optional[str] = Field(None, description="Attack name/label")
    target_id: str = Field(..., description="Target instance ID to attack")
    prepended_conversation: Optional[List[PrependedMessageRequest]] = Field(
        None, description="Messages to prepend (system prompts, branching context)"
    )
    converter_ids: Optional[List[str]] = Field(
        None, description="Converter instance IDs to apply to user messages"
    )


class CreateAttackResponse(BaseModel):
    """Response after creating an attack."""

    attack_id: str = Field(..., description="Unique attack identifier")
    name: Optional[str] = Field(None, description="Attack name/label")
    target_id: str = Field(..., description="Target instance ID")
    target_type: str = Field(..., description="Target type")
    outcome: Optional[str] = Field(None, description="Attack outcome (initially null)")
    prepended_conversation: List[Message] = Field(
        default_factory=list, description="Prepended messages (converted to Message format)"
    )
    created_at: datetime = Field(..., description="Attack creation timestamp")
    updated_at: datetime = Field(..., description="Last update timestamp")


# ============================================================================
# Update Attack
# ============================================================================


class UpdateAttackRequest(BaseModel):
    """Request to update an attack's outcome."""

    outcome: Literal["pending", "success", "failure"] = Field(
        ..., description="Updated attack outcome"
    )


# ============================================================================
# Send Message
# ============================================================================


class MessagePieceRequest(BaseModel):
    """A piece of content to send in a message."""

    data_type: str = Field(default="text", description="Data type: 'text', 'image', 'audio', etc.")
    content: str = Field(..., description="Content to send (text or base64 for media)")
    mime_type: Optional[str] = Field(None, description="MIME type for media content")


class SendMessageRequest(BaseModel):
    """Request to send a message within an attack."""

    pieces: List[MessagePieceRequest] = Field(..., description="Message pieces to send")
    converter_ids: Optional[List[str]] = Field(
        None, description="Converter instance IDs to apply (overrides attack-level)"
    )
    converters: Optional[List[InlineConverterConfig]] = Field(
        None, description="Inline converter definitions (for one-off use)"
    )


class SendMessageResponse(BaseModel):
    """Response after sending a message."""

    user_message: Message = Field(..., description="The user message that was sent")
    assistant_message: Message = Field(..., description="The assistant's response")
    attack_summary: AttackSummary = Field(..., description="Updated attack summary")
