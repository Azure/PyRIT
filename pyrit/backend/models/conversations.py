# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
Conversation-related request and response models.

These models align with PyRIT's MessagePiece and Message structures.
"""

from datetime import datetime
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field

from pyrit.models import PromptDataType, PromptResponseError


class ConverterConfig(BaseModel):
    """Configuration for a single converter."""

    class_name: str = Field(..., description="Converter class name (e.g., 'TranslationConverter')")
    module: str = Field(
        default="pyrit.prompt_converter",
        description="Module containing the converter class",
    )
    params: Optional[Dict[str, Any]] = Field(default=None, description="Constructor parameters")


# ============================================================================
# Conversation Creation
# ============================================================================


class CreateConversationRequest(BaseModel):
    """Request to create a new conversation."""

    target_class: str = Field(
        ...,
        description="Target class name (e.g., 'TextTarget', 'AzureOpenAIGPT4OChatTarget')",
    )
    target_params: Optional[Dict[str, Any]] = Field(
        None,
        description="Constructor parameters for the target",
    )
    labels: Optional[Dict[str, str]] = Field(None, description="Key-value labels for filtering")


class CreateConversationResponse(BaseModel):
    """Response after creating a conversation."""

    conversation_id: str = Field(..., description="Unique conversation identifier")
    target_identifier: Dict[str, Any] = Field(..., description="Target identifier (filtered)")
    labels: Optional[Dict[str, str]] = Field(None, description="Applied labels")
    created_at: datetime = Field(..., description="Creation timestamp")


# ============================================================================
# System Prompt
# ============================================================================


class SetSystemPromptRequest(BaseModel):
    """Request to set the system prompt for a conversation."""

    system_prompt: str = Field(..., description="The system prompt text")


class SystemPromptResponse(BaseModel):
    """Response containing the system prompt."""

    system_prompt: Optional[str] = Field(None, description="Current system prompt")
    piece_id: Optional[str] = Field(None, description="ID of the system prompt message piece")


# ============================================================================
# Converter Configuration
# ============================================================================


class SetConvertersRequest(BaseModel):
    """Request to set the converter chain for a conversation."""

    converters: List[ConverterConfig] = Field(..., description="Ordered list of converters")


class ConvertersResponse(BaseModel):
    """Response containing the converter chain."""

    converters: List[ConverterConfig] = Field(default_factory=list, description="Current converter chain")


# ============================================================================
# Message Pieces (aligned with MessagePiece)
# ============================================================================


class MessagePieceInput(BaseModel):
    """
    Input for a single message piece.

    Aligned with pyrit.models.MessagePiece fields.
    """

    original_value: Optional[str] = Field(None, description="Text content (for text type)")
    original_value_data_type: PromptDataType = Field(..., description="Data type of the content")
    file_name: Optional[str] = Field(None, description="Filename in multipart request (for file types)")
    converted_value: Optional[str] = Field(None, description="Pre-converted content (if pre_converted=true)")
    converted_value_data_type: Optional[PromptDataType] = Field(None, description="Data type after conversion")
    converter_identifiers: Optional[List[Dict[str, Any]]] = Field(
        None, description="Converters already applied (if pre_converted=true)"
    )


class MessagePieceResponse(BaseModel):
    """
    Response model for a single message piece.

    Aligned with pyrit.models.MessagePiece fields.
    """

    id: str = Field(..., description="Unique piece identifier (UUID)")
    original_value: str = Field(..., description="Original content or file path")
    original_value_data_type: PromptDataType = Field(..., description="Original data type")
    converted_value: str = Field(..., description="Converted content or file path")
    converted_value_data_type: PromptDataType = Field(..., description="Converted data type")
    converter_identifiers: List[Dict[str, Any]] = Field(
        default_factory=list, description="Applied converters with params"
    )
    response_error: Optional[PromptResponseError] = Field(None, description="Error type if any")
    timestamp: Optional[datetime] = Field(None, description="Piece timestamp")


# ============================================================================
# Messages
# ============================================================================

ChatMessageRole = Literal["system", "user", "assistant", "simulated_assistant", "tool", "developer"]


class MessageResponse(BaseModel):
    """Response model for a message (group of pieces with same sequence)."""

    sequence: int = Field(..., description="Sequence number in conversation")
    role: ChatMessageRole = Field(..., description="Message role")
    pieces: List[MessagePieceResponse] = Field(..., description="Message content pieces")
    timestamp: datetime = Field(..., description="Message timestamp")


class SendMessageRequest(BaseModel):
    """
    Request to send a message.

    Note: For file uploads, use multipart/form-data with 'pieces' as JSON
    and files attached with their filenames.
    """

    pieces: List[MessagePieceInput] = Field(..., description="Message content pieces")
    pre_converted: bool = Field(False, description="If true, skip converter chain")


class SendMessageResponse(BaseModel):
    """Response after sending a message."""

    user_message: MessageResponse = Field(..., description="The sent user message")
    assistant_message: Optional[MessageResponse] = Field(None, description="The assistant's response")


# ============================================================================
# Branch
# ============================================================================


class BranchConversationRequest(BaseModel):
    """Request to branch a conversation."""

    last_included_sequence: int = Field(..., description="Copy messages with sequence <= this value")


class BranchConversationResponse(BaseModel):
    """Response after branching a conversation."""

    conversation_id: str = Field(..., description="New conversation ID")
    branched_from: Dict[str, Any] = Field(..., description="Source conversation info")
    message_count: int = Field(..., description="Number of messages copied")
    created_at: datetime = Field(..., description="Branch creation timestamp")


# ============================================================================
# Full Conversation
# ============================================================================


class ConversationResponse(BaseModel):
    """Full conversation with all messages."""

    conversation_id: str = Field(..., description="Unique conversation identifier")
    target_identifier: Dict[str, Any] = Field(..., description="Target identifier (filtered)")
    labels: Optional[Dict[str, str]] = Field(None, description="Applied labels")
    converters: List[ConverterConfig] = Field(default_factory=list, description="Configured converters")
    created_at: datetime = Field(..., description="Creation timestamp")
    messages: List[MessageResponse] = Field(default_factory=list, description="All messages in order")
