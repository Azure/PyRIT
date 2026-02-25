# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
Attack-related request and response models.

All interactions in the UI are modeled as "attacks" - including manual conversations.
This is the attack-centric API design where every user interaction targets a model.
"""

from datetime import datetime
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field

from pyrit.backend.models.common import PaginationInfo
from pyrit.models import ChatMessageRole, PromptResponseError


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
    original_value_data_type: str = Field(
        default="text", description="Data type of the original value: 'text', 'image', 'audio', etc."
    )
    converted_value_data_type: str = Field(
        default="text", description="Data type of the converted value: 'text', 'image', 'audio', etc."
    )
    original_value: Optional[str] = Field(default=None, description="Original value before conversion")
    original_value_mime_type: Optional[str] = Field(default=None, description="MIME type of original value")
    converted_value: str = Field(..., description="Converted value (text or base64 for media)")
    converted_value_mime_type: Optional[str] = Field(default=None, description="MIME type of converted value")
    scores: list[Score] = Field(default_factory=list, description="Scores embedded in this piece")
    response_error: PromptResponseError = Field(
        default="none", description="Error status: none, processing, blocked, empty, unknown"
    )
    response_error_description: Optional[str] = Field(
        default=None, description="Description of the error if response_error is not 'none'"
    )


class Message(BaseModel):
    """A message within an attack."""

    turn_number: int = Field(..., description="Turn number in the conversation (1-indexed)")
    role: ChatMessageRole = Field(..., description="Message role")
    pieces: list[MessagePiece] = Field(..., description="Message pieces (multimodal support)")
    created_at: datetime = Field(..., description="Message creation timestamp")


# ============================================================================
# Attack Summary (List View)
# ============================================================================


class AttackSummary(BaseModel):
    """Summary view of an attack (for list views, omits full message content)."""

    conversation_id: str = Field(..., description="Unique attack identifier")
    attack_type: str = Field(..., description="Attack class name (e.g., 'CrescendoAttack', 'ManualAttack')")
    attack_specific_params: Optional[dict[str, Any]] = Field(None, description="Additional attack-specific parameters")
    target_unique_name: Optional[str] = Field(None, description="Unique name of the objective target")
    target_type: Optional[str] = Field(None, description="Target class name (e.g., 'OpenAIChatTarget')")
    converters: list[str] = Field(
        default_factory=list, description="Request converter class names applied in this attack"
    )
    outcome: Optional[Literal["undetermined", "success", "failure"]] = Field(
        None, description="Attack outcome (null if not yet determined)"
    )
    last_message_preview: Optional[str] = Field(
        None, description="Preview of the last message (truncated to ~100 chars)"
    )
    message_count: int = Field(0, description="Total number of messages in the attack")
    labels: dict[str, str] = Field(default_factory=dict, description="User-defined labels for filtering")
    created_at: datetime = Field(..., description="Attack creation timestamp")
    updated_at: datetime = Field(..., description="Last update timestamp")


# ============================================================================
# Attack Messages Response
# ============================================================================


class AttackMessagesResponse(BaseModel):
    """Response containing all messages for an attack."""

    conversation_id: str = Field(..., description="Attack identifier")
    messages: list[Message] = Field(default_factory=list, description="All messages in order")


# ============================================================================
# Attack List Response (Paginated)
# ============================================================================


class AttackListResponse(BaseModel):
    """Paginated response for listing attacks."""

    items: list[AttackSummary] = Field(..., description="List of attack summaries")
    pagination: PaginationInfo = Field(..., description="Pagination metadata")


class AttackOptionsResponse(BaseModel):
    """Response containing unique attack class names used across attacks."""

    attack_classes: list[str] = Field(
        ..., description="Sorted list of unique attack class names found in attack results"
    )


class ConverterOptionsResponse(BaseModel):
    """Response containing unique converter class names used across attacks."""

    converter_classes: list[str] = Field(
        ..., description="Sorted list of unique converter class names found in attack results"
    )


# ============================================================================
# Create Attack
# ============================================================================


# ============================================================================
# Message Input Models
# ============================================================================


class MessagePieceRequest(BaseModel):
    """A piece of content for a message."""

    data_type: str = Field(default="text", description="Data type: 'text', 'image', 'audio', etc.")
    original_value: str = Field(..., description="Original value (text or base64 for media)")
    converted_value: Optional[str] = Field(None, description="Converted value. If provided, bypasses converters.")
    mime_type: Optional[str] = Field(None, description="MIME type for media content")
    original_prompt_id: Optional[str] = Field(
        None,
        description="ID of the source piece when prepending from an existing conversation. "
        "Preserves lineage so the new piece traces back to the original.",
    )


class PrependedMessageRequest(BaseModel):
    """A message to prepend to the attack (for system prompt/branching)."""

    role: ChatMessageRole = Field(..., description="Message role")
    pieces: list[MessagePieceRequest] = Field(..., description="Message pieces (supports multimodal)", max_length=50)


class CreateAttackRequest(BaseModel):
    """Request to create a new attack."""

    name: Optional[str] = Field(None, description="Attack name/label")
    target_unique_name: str = Field(..., description="Target instance ID to attack")
    prepended_conversation: Optional[list[PrependedMessageRequest]] = Field(
        None, description="Messages to prepend (system prompts, branching context)", max_length=200
    )
    labels: Optional[dict[str, str]] = Field(None, description="User-defined labels for filtering")


class CreateAttackResponse(BaseModel):
    """Response after creating an attack."""

    conversation_id: str = Field(..., description="Unique attack identifier")
    created_at: datetime = Field(..., description="Attack creation timestamp")


# ============================================================================
# Update Attack
# ============================================================================


class UpdateAttackRequest(BaseModel):
    """Request to update an attack's outcome."""

    outcome: Literal["undetermined", "success", "failure"] = Field(..., description="Updated attack outcome")


# ============================================================================
# Add Message
# ============================================================================


class AddMessageRequest(BaseModel):
    """
    Request to add a message to an attack.

    If send=True (default for user role), the message is sent to the target
    and we wait for a response. If send=False, the message is just stored
    in memory without sending (useful for system messages, context injection).
    """

    role: ChatMessageRole = Field(default="user", description="Message role")
    pieces: list[MessagePieceRequest] = Field(..., description="Message pieces", max_length=50)
    send: bool = Field(
        default=True,
        description="If True, send to target and wait for response. If False, just store in memory.",
    )
    converter_ids: Optional[list[str]] = Field(
        None, description="Converter instance IDs to apply (overrides attack-level)"
    )


class AddMessageResponse(BaseModel):
    """
    Response after adding a message.

    Returns the attack metadata and all messages. If send=True was used, the new
    assistant response will be in the messages list. Check response_error
    on the assistant's message pieces if the target returned an error.
    """

    attack: AttackSummary = Field(..., description="Updated attack metadata")
    messages: AttackMessagesResponse = Field(..., description="All messages including new one(s)")
