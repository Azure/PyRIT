# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
Chat endpoints for conversation management
"""

from fastapi import APIRouter, HTTPException, File, UploadFile, Form
from typing import List, Optional
import base64
import os
import uuid
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

from pyrit.backend.models.requests import ChatRequest
from pyrit.backend.models.responses import ChatResponse, ConversationHistory
from pyrit.backend.services.chat_service import ChatService

router = APIRouter()
chat_service = ChatService()


@router.post("/chat", response_model=ChatResponse)
async def send_message(
    original_value: str = Form(...),  # The user's original text before conversion
    converted_value: Optional[str] = Form(None),  # The text after conversion (if converters applied)
    conversation_id: Optional[str] = Form(None),
    target_id: Optional[str] = Form(None),
    converter_identifiers: Optional[str] = Form(None),  # JSON string of PyRIT converter identifiers
    files: List[UploadFile] = File(default=[])
):
    """
    Send a message and get a response from the configured target.
    Supports multimodal input with file attachments.
    Accepts converter_identifiers from preview to properly track conversion history.
    """
    try:
        # Parse converter_identifiers if provided
        converter_ids = None
        if converter_identifiers:
            import json
            try:
                converter_ids = json.loads(converter_identifiers)
            except json.JSONDecodeError as je:
                logger.warning(f"Failed to parse converter_identifiers JSON: {je}")
        
        # Process uploaded files - save them locally
        attachments = []
        upload_dir = Path("/workspace/dbdata/prompt-memory-entries")
        
        logger.info(f"Received message request with {len(files)} files")
        
        for file in files:
            logger.info(f"Processing file: {file.filename}, type: {file.content_type}")
            content = await file.read()
            
            # Determine subdirectory based on content type
            if file.content_type and file.content_type.startswith('image/'):
                subdir = upload_dir / "images"
            elif file.content_type and file.content_type.startswith('audio/'):
                subdir = upload_dir / "audio"
            elif file.content_type and file.content_type.startswith('video/'):
                subdir = upload_dir / "videos"
            else:
                subdir = upload_dir / "urls"  # generic files
            
            subdir.mkdir(parents=True, exist_ok=True)
            
            # Generate unique filename
            file_ext = os.path.splitext(file.filename)[1]
            unique_filename = f"{uuid.uuid4()}{file_ext}"
            file_path = subdir / unique_filename
            
            # Save file
            with open(file_path, 'wb') as f:
                f.write(content)
            
            # Determine data type for PyRIT
            data_type = "text"
            if file.content_type:
                if file.content_type.startswith('image/'):
                    data_type = "image_path"
                elif file.content_type.startswith('audio/'):
                    data_type = "audio_path"
                elif file.content_type.startswith('video/'):
                    data_type = "video_path"
            
            attachments.append({
                'path': str(file_path),
                'name': file.filename,
                'content_type': file.content_type,
                'data_type': data_type,
                'size': len(content)
            })
        
        logger.info(f"Sending message with {len(attachments)} attachments to chat service")
        response = await chat_service.send_message(
            original_value=original_value,
            converted_value=converted_value,
            conversation_id=conversation_id,
            target_id=target_id,
            attachments=attachments if attachments else None,
            converter_identifiers=converter_ids,
        )
        return response
    except Exception as e:
        logger.exception(f"Failed to send message: {e}")
        # Include traceback in error response for debugging
        import traceback
        error_details = traceback.format_exc()
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to send message: {str(e)}\n\nTraceback:\n{error_details}"
        )


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
