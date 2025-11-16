# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
Response models for API endpoints
"""

from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime


class Message(BaseModel):
    """Individual message in a conversation"""

    role: str = Field(..., description="Message role (user/assistant/system)")
    content: str = Field(..., description="Message content")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Message timestamp")


class ChatResponse(BaseModel):
    """Response model for chat messages"""

    conversation_id: str = Field(..., description="Conversation identifier")
    message: str = Field(..., description="Response message from the target")
    role: str = Field(default="assistant", description="Role of the responder")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Response timestamp")
    target_id: Optional[str] = Field(None, description="Target that generated the response")

    class Config:
        json_schema_extra = {
            "example": {
                "conversation_id": "conv-123",
                "message": "I can help you test AI systems for various risks and vulnerabilities.",
                "role": "assistant",
                "timestamp": "2025-11-15T10:30:00Z",
                "target_id": "azure-openai-gpt4",
            }
        }


class ConversationHistory(BaseModel):
    """Conversation history with all messages"""

    conversation_id: str = Field(..., description="Conversation identifier")
    messages: List[Message] = Field(default_factory=list, description="List of messages in the conversation")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Conversation creation time")
    updated_at: datetime = Field(default_factory=datetime.utcnow, description="Last update time")
    target_id: Optional[str] = Field(None, description="Target used in this conversation")


class TargetInfo(BaseModel):
    """Information about a prompt target"""

    id: str = Field(..., description="Unique target identifier")
    name: str = Field(..., description="Human-readable target name")
    type: str = Field(..., description="Target type (e.g., azure_openai, openai)")
    description: str = Field(..., description="Target description")
    status: str = Field(default="available", description="Target availability status")

    class Config:
        json_schema_extra = {
            "example": {
                "id": "azure-openai-gpt4",
                "name": "Azure OpenAI GPT-4",
                "type": "azure_openai",
                "description": "Azure OpenAI GPT-4 chat model",
                "status": "available",
            }
        }
