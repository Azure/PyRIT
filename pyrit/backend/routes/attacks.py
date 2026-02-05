# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
Attack API routes.

All interactions are modeled as "attacks" - including manual conversations.
This is the attack-centric API design.
"""

from typing import Dict, List, Literal, Optional

from fastapi import APIRouter, HTTPException, Query, status

from pyrit.backend.models.attacks import (
    AddMessageRequest,
    AddMessageResponse,
    AttackListResponse,
    AttackMessagesResponse,
    AttackSummary,
    CreateAttackRequest,
    CreateAttackResponse,
    UpdateAttackRequest,
)
from pyrit.backend.models.common import ProblemDetail
from pyrit.backend.services.attack_service import get_attack_service

router = APIRouter(prefix="/attacks", tags=["attacks"])


def _parse_labels(label_params: Optional[List[str]]) -> Optional[Dict[str, str]]:
    """
    Parse label query params in 'key:value' format to a dict.

    Returns:
        Dict mapping label keys to values, or None if no valid labels.
    """
    if not label_params:
        return None
    labels = {}
    for param in label_params:
        if ":" in param:
            key, value = param.split(":", 1)
            labels[key.strip()] = value.strip()
    return labels if labels else None


@router.get(
    "",
    response_model=AttackListResponse,
)
async def list_attacks(
    target_id: Optional[str] = Query(None, description="Filter by target instance ID"),
    outcome: Optional[Literal["undetermined", "success", "failure"]] = Query(None, description="Filter by outcome"),
    name: Optional[str] = Query(None, description="Filter by attack name (substring match)"),
    label: Optional[List[str]] = Query(None, description="Filter by labels (format: key:value, repeatable)"),
    min_turns: Optional[int] = Query(None, ge=0, description="Filter by minimum executed turns"),
    max_turns: Optional[int] = Query(None, ge=0, description="Filter by maximum executed turns"),
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
    labels = _parse_labels(label)
    return await service.list_attacks(
        target_id=target_id,
        outcome=outcome,
        name=name,
        labels=labels,
        min_turns=min_turns,
        max_turns=max_turns,
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
    response_model=AttackSummary,
    responses={
        404: {"model": ProblemDetail, "description": "Attack not found"},
    },
)
async def get_attack(attack_id: str) -> AttackSummary:
    """
    Get attack details.

    Returns the attack metadata. Use GET /attacks/{id}/messages for messages.

    Returns:
        AttackSummary: Attack details.
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
    response_model=AttackSummary,
    responses={
        404: {"model": ProblemDetail, "description": "Attack not found"},
    },
)
async def update_attack(
    attack_id: str,
    request: UpdateAttackRequest,
) -> AttackSummary:
    """
    Update an attack's outcome.

    Used to mark attacks as success/failure/undetermined.

    Returns:
        AttackSummary: Updated attack details.
    """
    service = get_attack_service()

    attack = await service.update_attack(attack_id, request)
    if not attack:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Attack '{attack_id}' not found",
        )

    return attack


@router.get(
    "/{attack_id}/messages",
    response_model=AttackMessagesResponse,
    responses={
        404: {"model": ProblemDetail, "description": "Attack not found"},
    },
)
async def get_attack_messages(attack_id: str) -> AttackMessagesResponse:
    """
    Get all messages for an attack.

    Returns prepended conversation and all messages in order.

    Returns:
        AttackMessagesResponse: All messages for the attack.
    """
    service = get_attack_service()

    messages = await service.get_attack_messages(attack_id)
    if not messages:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Attack '{attack_id}' not found",
        )

    return messages


@router.post(
    "/{attack_id}/messages",
    response_model=AddMessageResponse,
    responses={
        404: {"model": ProblemDetail, "description": "Attack not found"},
        400: {"model": ProblemDetail, "description": "Message send failed"},
    },
)
async def add_message(
    attack_id: str,
    request: AddMessageRequest,
) -> AddMessageResponse:
    """
    Add a message to an attack.

    If send=True (default), sends the message to the target and waits for a response.
    If send=False, just stores the message in memory without sending (useful for
    system messages, context injection, or replaying assistant responses).

    Converters can be specified at three levels (in priority order):
    1. request.converter_ids - per-message converter instances
    2. request.converters - inline converter definitions
    3. attack.converter_ids - attack-level defaults

    Returns:
        AddMessageResponse: Updated attack with new message(s).
    """
    service = get_attack_service()

    try:
        return await service.add_message(attack_id, request)
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
            detail=f"Failed to add message: {str(e)}",
        )
