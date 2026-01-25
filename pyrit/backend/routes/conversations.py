# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
Conversation API routes.

Provides endpoints for managing interactive conversation sessions.
"""

from typing import List

from fastapi import APIRouter, HTTPException, status

from pyrit.backend.models.common import ProblemDetail
from pyrit.backend.models.conversations import (
    BranchConversationRequest,
    BranchConversationResponse,
    ConverterConfig,
    ConvertersResponse,
    CreateConversationRequest,
    CreateConversationResponse,
    MessageResponse,
    SendMessageRequest,
    SendMessageResponse,
    SetSystemPromptRequest,
    SystemPromptResponse,
)
from pyrit.backend.services import get_conversation_service

router = APIRouter(prefix="/conversations", tags=["conversations"])


@router.post(
    "",
    response_model=CreateConversationResponse,
    status_code=status.HTTP_201_CREATED,
    responses={
        400: {"model": ProblemDetail, "description": "Invalid request"},
        422: {"model": ProblemDetail, "description": "Validation error"},
    },
)
async def create_conversation(request: CreateConversationRequest) -> CreateConversationResponse:
    """
    Create a new conversation session.

    Establishes a new conversation with the specified target and optional
    system prompt and converters.

    Returns:
        CreateConversationResponse: The created conversation details.
    """
    service = get_conversation_service()

    try:
        return await service.create_conversation(request)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create conversation: {str(e)}",
        )


@router.get(
    "/{conversation_id}",
    response_model=List[MessageResponse],
    responses={
        404: {"model": ProblemDetail, "description": "Conversation not found"},
    },
)
async def get_conversation(conversation_id: str) -> List[MessageResponse]:
    """
    Get all messages in a conversation.

    Returns messages in sequence order.

    Returns:
        List[MessageResponse]: List of messages in the conversation.
    """
    service = get_conversation_service()

    state = await service.get_conversation(conversation_id)
    if not state:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Conversation {conversation_id} not found",
        )

    return await service.get_conversation_messages(conversation_id)


@router.post(
    "/{conversation_id}/messages",
    response_model=SendMessageResponse,
    responses={
        404: {"model": ProblemDetail, "description": "Conversation not found"},
        400: {"model": ProblemDetail, "description": "Message send failed"},
    },
)
async def send_message(
    conversation_id: str,
    request: SendMessageRequest,
) -> SendMessageResponse:
    """
    Send a message in a conversation.

    Sends the user message to the target, applies converters, and returns
    both the sent message and assistant response(s).

    Returns:
        SendMessageResponse: The sent message and assistant response.
    """
    service = get_conversation_service()

    state = await service.get_conversation(conversation_id)
    if not state:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Conversation {conversation_id} not found",
        )

    try:
        return await service.send_message(conversation_id, request)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to send message: {str(e)}",
        )


@router.get(
    "/{conversation_id}/system-prompt",
    response_model=SystemPromptResponse,
    responses={
        404: {"model": ProblemDetail, "description": "Conversation not found"},
    },
)
async def get_system_prompt(conversation_id: str) -> SystemPromptResponse:
    """
    Get the current system prompt for a conversation.

    Returns:
        SystemPromptResponse: The current system prompt.
    """
    service = get_conversation_service()

    state = await service.get_conversation(conversation_id)
    if not state:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Conversation {conversation_id} not found",
        )

    return SystemPromptResponse(
        system_prompt=state.system_prompt,
        piece_id=None,  # System prompts stored in state, not as MessagePiece
    )


@router.put(
    "/{conversation_id}/system-prompt",
    response_model=SystemPromptResponse,
    responses={
        404: {"model": ProblemDetail, "description": "Conversation not found"},
    },
)
async def update_system_prompt(
    conversation_id: str,
    request: SetSystemPromptRequest,
) -> SystemPromptResponse:
    """
    Update the system prompt for a conversation.

    Takes effect for subsequent messages.

    Returns:
        SystemPromptResponse: The updated system prompt.
    """
    service = get_conversation_service()

    state = await service.get_conversation(conversation_id)
    if not state:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Conversation {conversation_id} not found",
        )

    await service.update_system_prompt(conversation_id, request.system_prompt)

    return SystemPromptResponse(
        system_prompt=request.system_prompt,
        piece_id=None,
    )


@router.get(
    "/{conversation_id}/converters",
    response_model=ConvertersResponse,
    responses={
        404: {"model": ProblemDetail, "description": "Conversation not found"},
    },
)
async def get_converters(conversation_id: str) -> ConvertersResponse:
    """
    Get the current converters for a conversation.

    Returns:
        ConvertersResponse: The current converter configurations.
    """
    service = get_conversation_service()

    state = await service.get_conversation(conversation_id)
    if not state:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Conversation {conversation_id} not found",
        )

    return ConvertersResponse(
        converters=state.converters,
    )


@router.put(
    "/{conversation_id}/converters",
    response_model=ConvertersResponse,
    responses={
        404: {"model": ProblemDetail, "description": "Conversation not found"},
        400: {"model": ProblemDetail, "description": "Invalid converter configuration"},
    },
)
async def update_converters(
    conversation_id: str,
    converters: List[ConverterConfig],
) -> ConvertersResponse:
    """
    Update the converters for a conversation.

    Replaces all current converters with the provided list.
    Takes effect for subsequent messages.

    Returns:
        ConvertersResponse: The updated converter configurations.
    """
    service = get_conversation_service()

    state = await service.get_conversation(conversation_id)
    if not state:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Conversation {conversation_id} not found",
        )

    try:
        await service.update_converters(conversation_id, converters)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )

    return ConvertersResponse(
        converters=converters,
    )


@router.post(
    "/{conversation_id}/branch",
    response_model=BranchConversationResponse,
    status_code=status.HTTP_201_CREATED,
    responses={
        404: {"model": ProblemDetail, "description": "Conversation not found"},
        400: {"model": ProblemDetail, "description": "Invalid branch request"},
    },
)
async def branch_conversation(
    conversation_id: str,
    request: BranchConversationRequest,
) -> BranchConversationResponse:
    """
    Branch a conversation from a specific point.

    Creates a new conversation with messages copied up to and including
    the specified sequence number. The original conversation is unchanged.

    Returns:
        BranchConversationResponse: The new branched conversation details.
    """
    service = get_conversation_service()

    state = await service.get_conversation(conversation_id)
    if not state:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Conversation {conversation_id} not found",
        )

    try:
        return await service.branch_conversation(conversation_id, request)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )


@router.delete(
    "/{conversation_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    responses={
        404: {"model": ProblemDetail, "description": "Conversation not found"},
    },
)
async def delete_conversation(conversation_id: str) -> None:
    """
    Delete a conversation session.

    Cleans up in-memory resources. Messages remain in memory database.
    """
    service = get_conversation_service()

    state = await service.get_conversation(conversation_id)
    if not state:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Conversation {conversation_id} not found",
        )

    service.cleanup_conversation(conversation_id)
