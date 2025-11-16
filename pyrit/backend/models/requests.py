# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
Request models for API endpoints
"""

from pydantic import BaseModel, Field
from typing import Optional


class ChatRequest(BaseModel):
    """Request model for sending a chat message"""

    message: str = Field(..., description="The message to send", min_length=1)
    conversation_id: Optional[str] = Field(None, description="Optional conversation ID to continue an existing chat")
    target_id: Optional[str] = Field(None, description="Optional target ID to use for this message")

    class Config:
        json_schema_extra = {
            "example": {
                "message": "Hello, how can you help me test my AI system?",
                "conversation_id": "conv-123",
                "target_id": "azure-openai-gpt4",
            }
        }
