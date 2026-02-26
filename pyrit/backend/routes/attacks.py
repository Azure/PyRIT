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
    AddMessageRequest,
    AddMessageResponse,
    AttackListResponse,
    AttackMessagesResponse,
    AttackOptionsResponse,
    AttackSummary,
    ConverterOptionsResponse,
    CreateAttackRequest,
    CreateAttackResponse,
    UpdateAttackRequest,
)
from pyrit.backend.models.common import ProblemDetail
from pyrit.backend.services.attack_service import get_attack_service

router = APIRouter(prefix="/attacks", tags=["attacks"])


def _parse_labels(label_params: Optional[list[str]]) -> Optional[dict[str, str]]:
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
    attack_class: Optional[str] = Query(None, description="Filter by exact attack class name"),
    converter_classes: Optional[list[str]] = Query(
        None,
        description="Filter by converter class names (repeatable, AND logic). Pass empty to match no-converter attacks.",
    ),
    outcome: Optional[Literal["undetermined", "success", "failure"]] = Query(None, description="Filter by outcome"),
    label: Optional[list[str]] = Query(None, description="Filter by labels (format: key:value, repeatable)"),
    min_turns: Optional[int] = Query(None, ge=0, description="Filter by minimum executed turns"),
    max_turns: Optional[int] = Query(None, ge=0, description="Filter by maximum executed turns"),
    limit: int = Query(20, ge=1, le=100, description="Maximum items per page"),
    cursor: Optional[str] = Query(None, description="Pagination cursor (conversation_id)"),
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
    return await service.list_attacks_async(
        attack_class=attack_class,
        converter_classes=converter_classes,
        outcome=outcome,
        labels=labels,
        min_turns=min_turns,
        max_turns=max_turns,
        limit=limit,
        cursor=cursor,
    )


@router.get(
    "/attack-options",
    response_model=AttackOptionsResponse,
)
async def get_attack_options() -> AttackOptionsResponse:
    """
    Get unique attack class names used across all attacks.

    Returns all attack class names found in stored attack results.
    Useful for populating attack type filter dropdowns in the GUI.

    Returns:
        AttackOptionsResponse: Sorted list of unique attack class names.
    """
    service = get_attack_service()
    class_names = await service.get_attack_options_async()
    return AttackOptionsResponse(attack_classes=class_names)


@router.get(
    "/converter-options",
    response_model=ConverterOptionsResponse,
)
async def get_converter_options() -> ConverterOptionsResponse:
    """
    Get unique converter class names used across all attacks.

    Returns all converter class names found in stored attack results.
    Useful for populating converter filter dropdowns in the GUI.

    Returns:
        ConverterOptionsResponse: Sorted list of unique converter class names.
    """
    service = get_attack_service()
    class_names = await service.get_converter_options_async()
    return ConverterOptionsResponse(converter_classes=class_names)


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
        return await service.create_attack_async(request=request)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e),
        ) from e


@router.get(
    "/{conversation_id}",
    response_model=AttackSummary,
    responses={
        404: {"model": ProblemDetail, "description": "Attack not found"},
    },
)
async def get_attack(conversation_id: str) -> AttackSummary:
    """
    Get attack details.

    Returns the attack metadata. Use GET /attacks/{id}/messages for messages.

    Returns:
        AttackSummary: Attack details.
    """
    service = get_attack_service()

    attack = await service.get_attack_async(conversation_id=conversation_id)
    if not attack:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Attack '{conversation_id}' not found",
        )

    return attack


@router.patch(
    "/{conversation_id}",
    response_model=AttackSummary,
    responses={
        404: {"model": ProblemDetail, "description": "Attack not found"},
    },
)
async def update_attack(
    conversation_id: str,
    request: UpdateAttackRequest,
) -> AttackSummary:
    """
    Update an attack's outcome.

    Used to mark attacks as success/failure/undetermined.

    Returns:
        AttackSummary: Updated attack details.
    """
    service = get_attack_service()

    attack = await service.update_attack_async(conversation_id=conversation_id, request=request)
    if not attack:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Attack '{conversation_id}' not found",
        )

    return attack


@router.get(
    "/{conversation_id}/messages",
    response_model=AttackMessagesResponse,
    responses={
        404: {"model": ProblemDetail, "description": "Attack not found"},
    },
)
async def get_attack_messages(conversation_id: str) -> AttackMessagesResponse:
    """
    Get all messages for an attack.

    Returns prepended conversation and all messages in order.

    Returns:
        AttackMessagesResponse: All messages for the attack.
    """
    service = get_attack_service()

    messages = await service.get_attack_messages_async(conversation_id=conversation_id)
    if not messages:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Attack '{conversation_id}' not found",
        )

    return messages


@router.post(
    "/{conversation_id}/messages",
    response_model=AddMessageResponse,
    responses={
        404: {"model": ProblemDetail, "description": "Attack not found"},
        400: {"model": ProblemDetail, "description": "Message send failed"},
    },
)
async def add_message(
    conversation_id: str,
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
        return await service.add_message_async(conversation_id=conversation_id, request=request)
    except ValueError as e:
        error_msg = str(e)
        if "not found" in error_msg.lower():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=error_msg,
            ) from e
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=error_msg,
        ) from e
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to add message: {str(e)}",
        ) from e
