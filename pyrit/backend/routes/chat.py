# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
Chat endpoints for conversation management
"""

from fastapi import APIRouter, HTTPException
from typing import List

from pyrit.backend.models.requests import ChatRequest
from pyrit.backend.models.responses import ChatResponse, ConversationHistory
from pyrit.backend.services.chat_service import ChatService

router = APIRouter()
chat_service = ChatService()


@router.post("/chat", response_model=ChatResponse)
async def send_message(request: ChatRequest):
    """
    Send a message and get a response from the configured target
    """
    try:
        response = await chat_service.send_message(
            message=request.message,
            conversation_id=request.conversation_id,
            target_id=request.target_id,
        )
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to send message: {str(e)}")


@router.get("/chat/conversations", response_model=List[ConversationHistory])
async def get_conversations():
    """
    Get all conversation histories
    """
    try:
        conversations = await chat_service.get_conversations()
        return conversations
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve conversations: {str(e)}")


@router.get("/chat/conversations/{conversation_id}", response_model=ConversationHistory)
async def get_conversation(conversation_id: str):
    """
    Get a specific conversation by ID
    """
    try:
        conversation = await chat_service.get_conversation(conversation_id)
        if not conversation:
            raise HTTPException(status_code=404, detail="Conversation not found")
        return conversation
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve conversation: {str(e)}")


@router.delete("/chat/conversations/{conversation_id}")
async def delete_conversation(conversation_id: str):
    """
    Delete a conversation
    """
    try:
        success = await chat_service.delete_conversation(conversation_id)
        if not success:
            raise HTTPException(status_code=404, detail="Conversation not found")
        return {"status": "deleted", "conversation_id": conversation_id}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete conversation: {str(e)}")
