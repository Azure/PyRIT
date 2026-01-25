# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
Memory API routes.

Provides endpoints for querying stored data with pagination.
"""

from datetime import datetime
from typing import List, Optional

from fastapi import APIRouter, Query

from pyrit.backend.models.common import PaginatedResponse
from pyrit.backend.models.memory import (
    AttackResultQueryResponse,
    MessageQueryResponse,
    ScenarioResultQueryResponse,
    ScoreQueryResponse,
    SeedQueryResponse,
)
from pyrit.backend.services import get_memory_service

router = APIRouter(prefix="/memory", tags=["memory"])


@router.get(
    "/messages",
    response_model=PaginatedResponse[MessageQueryResponse],
)
async def query_messages(
    conversation_id: Optional[str] = Query(None, description="Filter by conversation ID"),
    role: Optional[str] = Query(None, description="Filter by role (user/assistant/system)"),
    data_type: Optional[str] = Query(None, description="Filter by data type (text/image_path/audio_path)"),
    harm_category: Optional[List[str]] = Query(None, description="Filter by harm categories"),
    response_error: Optional[str] = Query(None, description="Filter by response error type"),
    start_time: Optional[datetime] = Query(None, description="Messages after this time"),
    end_time: Optional[datetime] = Query(None, description="Messages before this time"),
    limit: int = Query(50, ge=1, le=200, description="Maximum results per page"),
    cursor: Optional[str] = Query(None, description="Pagination cursor"),
) -> PaginatedResponse[MessageQueryResponse]:
    """
    Query message pieces with pagination.

    Returns messages matching the specified filters, ordered by timestamp descending.
    Use cursor for pagination through large result sets.

    Returns:
        PaginatedResponse[MessageQueryResponse]: Paginated list of messages.
    """
    service = get_memory_service()

    return await service.get_messages(
        conversation_id=conversation_id,
        role=role,
        harm_categories=harm_category,
        data_type=data_type,
        response_error=response_error,
        start_time=start_time,
        end_time=end_time,
        limit=limit,
        cursor=cursor,
    )


@router.get(
    "/scores",
    response_model=PaginatedResponse[ScoreQueryResponse],
)
async def query_scores(
    message_id: Optional[str] = Query(None, description="Filter by message piece ID"),
    conversation_id: Optional[str] = Query(None, description="Filter by conversation ID"),
    score_type: Optional[str] = Query(None, description="Filter by score type"),
    scorer_type: Optional[str] = Query(None, description="Filter by scorer class name"),
    start_time: Optional[datetime] = Query(None, description="Scores after this time"),
    end_time: Optional[datetime] = Query(None, description="Scores before this time"),
    limit: int = Query(50, ge=1, le=200, description="Maximum results per page"),
    cursor: Optional[str] = Query(None, description="Pagination cursor"),
) -> PaginatedResponse[ScoreQueryResponse]:
    """
    Query scores with pagination.

    Returns scores matching the specified filters, ordered by timestamp descending.

    Returns:
        PaginatedResponse[ScoreQueryResponse]: Paginated list of scores.
    """
    service = get_memory_service()

    return await service.get_scores(
        message_id=message_id,
        conversation_id=conversation_id,
        score_type=score_type,
        scorer_type=scorer_type,
        start_time=start_time,
        end_time=end_time,
        limit=limit,
        cursor=cursor,
    )


@router.get(
    "/attack-results",
    response_model=PaginatedResponse[AttackResultQueryResponse],
)
async def query_attack_results(
    conversation_id: Optional[str] = Query(None, description="Filter by conversation ID"),
    outcome: Optional[str] = Query(None, description="Filter by outcome"),
    attack_type: Optional[str] = Query(None, description="Filter by attack class name"),
    objective: Optional[str] = Query(None, description="Search by objective text"),
    min_turns: Optional[int] = Query(None, ge=1, description="Minimum executed turns"),
    max_turns: Optional[int] = Query(None, ge=1, description="Maximum executed turns"),
    start_time: Optional[datetime] = Query(None, description="Results after this time"),
    end_time: Optional[datetime] = Query(None, description="Results before this time"),
    limit: int = Query(50, ge=1, le=200, description="Maximum results per page"),
    cursor: Optional[str] = Query(None, description="Pagination cursor"),
) -> PaginatedResponse[AttackResultQueryResponse]:
    """
    Query attack results with pagination.

    Returns attack results matching the specified filters, ordered by timestamp descending.

    Returns:
        PaginatedResponse[AttackResultQueryResponse]: Paginated list of attack results.
    """
    service = get_memory_service()

    return await service.get_attack_results(
        conversation_id=conversation_id,
        outcome=outcome,
        attack_type=attack_type,
        objective=objective,
        min_turns=min_turns,
        max_turns=max_turns,
        start_time=start_time,
        end_time=end_time,
        limit=limit,
        cursor=cursor,
    )


@router.get(
    "/scenario-results",
    response_model=PaginatedResponse[ScenarioResultQueryResponse],
)
async def query_scenario_results(
    scenario_name: Optional[str] = Query(None, description="Filter by scenario name"),
    run_state: Optional[str] = Query(None, description="Filter by run state"),
    start_time: Optional[datetime] = Query(None, description="Results after this time"),
    end_time: Optional[datetime] = Query(None, description="Results before this time"),
    limit: int = Query(50, ge=1, le=200, description="Maximum results per page"),
    cursor: Optional[str] = Query(None, description="Pagination cursor"),
) -> PaginatedResponse[ScenarioResultQueryResponse]:
    """
    Query scenario results with pagination.

    Returns scenario results matching the specified filters, ordered by timestamp descending.

    Returns:
        PaginatedResponse[ScenarioResultQueryResponse]: Paginated list of scenario results.
    """
    service = get_memory_service()

    return await service.get_scenario_results(
        scenario_name=scenario_name,
        run_state=run_state,
        start_time=start_time,
        end_time=end_time,
        limit=limit,
        cursor=cursor,
    )


@router.get(
    "/seeds",
    response_model=PaginatedResponse[SeedQueryResponse],
)
async def query_seeds(
    dataset_name: Optional[str] = Query(None, description="Filter by dataset name"),
    seed_type: Optional[str] = Query(None, description="Filter by seed type"),
    harm_category: Optional[List[str]] = Query(None, description="Filter by harm categories"),
    data_type: Optional[str] = Query(None, description="Filter by data type"),
    search: Optional[str] = Query(None, description="Search in seed value text"),
    limit: int = Query(50, ge=1, le=200, description="Maximum results per page"),
    cursor: Optional[str] = Query(None, description="Pagination cursor"),
) -> PaginatedResponse[SeedQueryResponse]:
    """
    Query seeds with pagination.

    Returns seeds matching the specified filters, ordered by date_added descending.

    Returns:
        PaginatedResponse[SeedQueryResponse]: Paginated list of seeds.
    """
    service = get_memory_service()

    return await service.get_seeds(
        dataset_name=dataset_name,
        seed_type=seed_type,
        harm_categories=harm_category,
        data_type=data_type,
        search=search,
        limit=limit,
        cursor=cursor,
    )
