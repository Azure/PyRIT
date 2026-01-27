# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
Attack API routes.

All interactions are modeled as "attacks" - including manual conversations.
This is the attack-centric API design.
"""

from typing import Literal, Optional

from fastapi import APIRouter, HTTPException, Query, status

from pyrit.backend.models.attacks import (
    AttackDetail,
    AttackListResponse,
    CreateAttackRequest,
    CreateAttackResponse,
    SendMessageRequest,
    SendMessageResponse,
    UpdateAttackRequest,
)
from pyrit.backend.models.common import ProblemDetail
from pyrit.backend.services.attack_service import get_attack_service

router = APIRouter(prefix="/attacks", tags=["attacks"])


@router.get(
    "",
    response_model=AttackListResponse,
)
async def list_attacks(
    target_id: Optional[str] = Query(None, description="Filter by target instance ID"),
    outcome: Optional[Literal["pending", "success", "failure"]] = Query(None, description="Filter by outcome"),
    limit: int = Query(20, ge=1, le=100, description="Maximum items per page"),
    cursor: Optional[str] = Query(None, description="Pagination cursor (attack_id)"),
) -> AttackListResponse:
    """
    List attacks with optional filtering and pagination.

    Returns attack summaries (not full message content).
    Use GET /attacks/{id} for full details.

    Returns:
        AttackListResponse: Paginated list of attack summaries.
    """
    service = get_attack_service()
    return await service.list_attacks(
        target_id=target_id,
        outcome=outcome,
        limit=limit,
        cursor=cursor,
    )


@router.post(
    "",
    response_model=CreateAttackResponse,
    status_code=status.HTTP_201_CREATED,
    responses={
        400: {"model": ProblemDetail, "description": "Invalid request"},
        404: {"model": ProblemDetail, "description": "Target or converter not found"},
        422: {"model": ProblemDetail, "description": "Validation error"},
    },
)
async def create_attack(request: CreateAttackRequest) -> CreateAttackResponse:
    """
    Create a new attack.

    Establishes a new attack session with the specified target.
    Optionally include prepended_conversation for system prompts or branching context.

    Returns:
        CreateAttackResponse: The created attack details.
    """
    service = get_attack_service()

    try:
        return await service.create_attack(request)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e),
        )


@router.get(
    "/{attack_id}",
    response_model=AttackDetail,
    responses={
        404: {"model": ProblemDetail, "description": "Attack not found"},
    },
)
async def get_attack(attack_id: str) -> AttackDetail:
    """
    Get attack details including all messages.

    Returns the full attack with prepended_conversation and all messages.

    Returns:
        AttackDetail: Full attack details with messages.
    """
    service = get_attack_service()

    attack = await service.get_attack(attack_id)
    if not attack:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Attack '{attack_id}' not found",
        )

    return attack


@router.patch(
    "/{attack_id}",
    response_model=AttackDetail,
    responses={
        404: {"model": ProblemDetail, "description": "Attack not found"},
    },
)
async def update_attack(
    attack_id: str,
    request: UpdateAttackRequest,
) -> AttackDetail:
    """
    Update an attack's outcome.

    Used to mark attacks as success/failure/pending.

    Returns:
        AttackDetail: Updated attack details.
    """
    service = get_attack_service()

    attack = await service.update_attack(attack_id, request)
    if not attack:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Attack '{attack_id}' not found",
        )

    return attack


@router.post(
    "/{attack_id}/messages",
    response_model=SendMessageResponse,
    responses={
        404: {"model": ProblemDetail, "description": "Attack not found"},
        400: {"model": ProblemDetail, "description": "Message send failed"},
    },
)
async def send_message(
    attack_id: str,
    request: SendMessageRequest,
) -> SendMessageResponse:
    """
    Send a message in an attack.

    Sends the user message to the target, applies converters, and returns
    both the user message and assistant response.

    Converters can be specified at three levels (in priority order):
    1. request.converter_ids - per-message converter instances
    2. request.converters - inline converter definitions
    3. attack.converter_ids - attack-level defaults

    Returns:
        SendMessageResponse: The sent message and assistant response.
    """
    service = get_attack_service()

    try:
        return await service.send_message(attack_id, request)
    except ValueError as e:
        error_msg = str(e)
        if "not found" in error_msg.lower():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=error_msg,
            )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=error_msg,
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to send message: {str(e)}",
        )


@router.delete(
    "/{attack_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    responses={
        404: {"model": ProblemDetail, "description": "Attack not found"},
    },
)
async def delete_attack(attack_id: str) -> None:
    """
    Delete an attack.

    Removes the attack and all associated messages.
    """
    service = get_attack_service()

    deleted = await service.delete_attack(attack_id)
    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Attack '{attack_id}' not found",
        )
