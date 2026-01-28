# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
Target API routes.

Provides endpoints for managing target instances.
Target types are set at app startup via initializers - you cannot add new types at runtime.
"""

from typing import Literal, Optional

from fastapi import APIRouter, HTTPException, Query, status

from pyrit.backend.models.common import ProblemDetail
from pyrit.backend.models.targets import (
    CreateTargetRequest,
    CreateTargetResponse,
    TargetInstance,
    TargetListResponse,
)
from pyrit.backend.services.target_service import get_target_service

router = APIRouter(prefix="/targets", tags=["targets"])


@router.get(
    "",
    response_model=TargetListResponse,
)
async def list_targets(
    source: Optional[Literal["initializer", "user"]] = Query(
        None, description="Filter by source (initializer or user)"
    ),
) -> TargetListResponse:
    """
    List target instances.

    Returns all registered target instances. Use source filter to distinguish
    between initializer-created (startup) and user-created (API) targets.

    Returns:
        TargetListResponse: List of target instances.
    """
    service = get_target_service()
    return await service.list_targets(source=source)


@router.post(
    "",
    response_model=CreateTargetResponse,
    status_code=status.HTTP_201_CREATED,
    responses={
        400: {"model": ProblemDetail, "description": "Invalid target type or parameters"},
    },
)
async def create_target(request: CreateTargetRequest) -> CreateTargetResponse:
    """
    Create a new target instance.

    Instantiates a target with the given type and parameters.
    The target becomes available for use in attacks.

    Note: Sensitive parameters (API keys, tokens) are filtered from the response.

    Returns:
        CreateTargetResponse: The created target instance details.
    """
    service = get_target_service()

    try:
        return await service.create_target(request)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create target: {str(e)}",
        )


@router.get(
    "/{target_id}",
    response_model=TargetInstance,
    responses={
        404: {"model": ProblemDetail, "description": "Target not found"},
    },
)
async def get_target(target_id: str) -> TargetInstance:
    """
    Get a target instance by ID.

    Returns:
        TargetInstance: The target instance details.
    """
    service = get_target_service()

    target = await service.get_target(target_id)
    if not target:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Target '{target_id}' not found",
        )

    return target
