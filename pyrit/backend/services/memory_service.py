# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
Memory service for API access to stored data.

Wraps CentralMemory with pagination and filtering for API endpoints.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from pyrit.backend.models.common import PaginatedResponse, PaginationInfo, filter_sensitive_fields
from pyrit.backend.models.memory import (
    AttackResultQueryResponse,
    MessageQueryResponse,
    ScenarioResultQueryResponse,
    ScoreQueryResponse,
    SeedQueryResponse,
)
from pyrit.memory import CentralMemory
from pyrit.models.seeds import SeedObjective, SeedSimulatedConversation


def _parse_cursor(cursor: Optional[str]) -> Tuple[Optional[datetime], Optional[str]]:
    """
    Parse a cursor string into timestamp and ID components.

    Cursor format: {ISO8601_timestamp}_{record_id}

    Returns:
        Tuple[Optional[datetime], Optional[str]]: Parsed timestamp and record ID.
    """
    if not cursor:
        return None, None

    try:
        parts = cursor.rsplit("_", 1)
        if len(parts) != 2:
            return None, None
        timestamp_str, record_id = parts
        timestamp = datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
        return timestamp, record_id
    except (ValueError, AttributeError):
        return None, None


def _build_cursor(timestamp: datetime, record_id: str) -> str:
    """
    Build a cursor string from timestamp and ID.

    Returns:
        str: Cursor string for pagination.
    """
    return f"{timestamp.isoformat()}_{record_id}"


class MemoryService:
    """Service for querying memory with pagination support."""

    def __init__(self) -> None:
        """Initialize the memory service."""
        self._memory = CentralMemory.get_memory_instance()

    async def get_messages(
        self,
        *,
        conversation_id: Optional[str] = None,
        role: Optional[str] = None,
        labels: Optional[Dict[str, str]] = None,
        harm_categories: Optional[List[str]] = None,
        data_type: Optional[str] = None,
        response_error: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 50,
        cursor: Optional[str] = None,
    ) -> PaginatedResponse[MessageQueryResponse]:
        """
        Query message pieces with pagination.

        Args:
            conversation_id: Filter by conversation.
            role: Filter by message role.
            labels: Filter by labels.
            harm_categories: Filter by harm categories (not supported in current API).
            data_type: Filter by data type.
            response_error: Filter by response error type (not supported in current API).
            start_time: Messages after this time.
            end_time: Messages before this time.
            limit: Maximum results per page.
            cursor: Pagination cursor.

        Returns:
            Paginated list of messages.
        """
        # Parse cursor for pagination
        cursor_time, cursor_id = _parse_cursor(cursor)

        # Query memory - use supported parameters only
        pieces = self._memory.get_message_pieces(
            conversation_id=conversation_id,
            role=role,
            labels=labels,
            data_type=data_type,
            sent_after=cursor_time or start_time,
            sent_before=end_time,
        )

        # Apply start_time filter if provided and no cursor
        if start_time and not cursor_time:
            pieces = [p for p in pieces if p.timestamp and p.timestamp >= start_time]

        # Sort by timestamp descending
        pieces = sorted(pieces, key=lambda p: p.timestamp or datetime.min, reverse=True)

        # Apply limit + 1 to check for more
        has_more = len(pieces) > limit
        pieces = pieces[:limit]

        # Build response items
        items = []
        for piece in pieces:
            items.append(
                MessageQueryResponse(
                    id=str(piece.id),
                    conversation_id=piece.conversation_id,
                    sequence=piece.sequence,
                    role=piece.role,
                    original_value=piece.original_value,
                    original_value_data_type=piece.original_value_data_type,
                    converted_value=piece.converted_value,
                    converted_value_data_type=piece.converted_value_data_type,
                    converter_identifiers=piece.converter_identifiers or [],
                    target_identifier=filter_sensitive_fields(piece.prompt_target_identifier or {}),
                    labels=piece.labels,
                    response_error=piece.response_error,
                    timestamp=piece.timestamp,
                )
            )

        # Build pagination info
        next_cursor = None
        if has_more and pieces:
            last_piece = pieces[-1]
            next_cursor = _build_cursor(last_piece.timestamp, str(last_piece.id))

        prev_cursor = None
        if cursor and pieces:
            first_piece = pieces[0]
            prev_cursor = _build_cursor(first_piece.timestamp, str(first_piece.id))

        return PaginatedResponse(
            items=items,
            pagination=PaginationInfo(
                limit=limit,
                has_more=has_more,
                next_cursor=next_cursor,
                prev_cursor=prev_cursor,
            ),
        )

    async def get_scores(
        self,
        *,
        message_id: Optional[str] = None,
        conversation_id: Optional[str] = None,
        score_type: Optional[str] = None,
        scorer_type: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 50,
        cursor: Optional[str] = None,
    ) -> PaginatedResponse[ScoreQueryResponse]:
        """
        Query scores with pagination.

        Returns:
            PaginatedResponse[ScoreQueryResponse]: Paginated list of scores.
        """
        scores = self._memory.get_scores(
            score_type=score_type,
        )

        # Apply additional filters
        if message_id:
            scores = [s for s in scores if str(s.message_piece_id) == message_id]

        if scorer_type:
            scores = [
                s
                for s in scores
                if s.scorer_class_identifier and s.scorer_class_identifier.get("__type__") == scorer_type
            ]

        if start_time:
            scores = [s for s in scores if s.timestamp and s.timestamp >= start_time]
        if end_time:
            scores = [s for s in scores if s.timestamp and s.timestamp <= end_time]

        # Sort and paginate
        scores = sorted(scores, key=lambda s: s.timestamp or datetime.min, reverse=True)
        has_more = len(scores) > limit
        scores = scores[:limit]

        items = []
        for score in scores:
            items.append(
                ScoreQueryResponse(
                    id=str(score.id),
                    message_piece_id=str(score.message_piece_id),
                    score_value=score.score_value,
                    score_value_description=score.score_value_description or "",
                    score_type=score.score_type,
                    score_category=score.score_category,
                    score_rationale=score.score_rationale or "",
                    scorer_identifier=filter_sensitive_fields(score.scorer_class_identifier or {}),
                    objective=score.objective,
                    timestamp=score.timestamp,
                )
            )

        next_cursor = None
        if has_more and scores:
            last = scores[-1]
            next_cursor = _build_cursor(last.timestamp, str(last.id))

        return PaginatedResponse(
            items=items,
            pagination=PaginationInfo(limit=limit, has_more=has_more, next_cursor=next_cursor, prev_cursor=None),
        )

    async def get_attack_results(
        self,
        *,
        conversation_id: Optional[str] = None,
        outcome: Optional[str] = None,
        attack_type: Optional[str] = None,
        objective: Optional[str] = None,
        min_turns: Optional[int] = None,
        max_turns: Optional[int] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 50,
        cursor: Optional[str] = None,
    ) -> PaginatedResponse[AttackResultQueryResponse]:
        """
        Query attack results with pagination.

        Returns:
            PaginatedResponse[AttackResultQueryResponse]: Paginated list of attack results.
        """
        results = self._memory.get_attack_results(
            conversation_id=conversation_id,
            outcome=outcome,
            objective=objective,
        )

        # Apply additional filters
        if attack_type:
            results = [r for r in results if r.attack_identifier and r.attack_identifier.get("__type__") == attack_type]

        if min_turns:
            results = [r for r in results if r.executed_turns >= min_turns]
        if max_turns:
            results = [r for r in results if r.executed_turns <= max_turns]

        # Note: AttackResult doesn't have timestamp field - skip time filtering
        # Sort by executed_turns as a proxy for recency
        results_list = list(results)
        has_more = len(results_list) > limit
        results_list = results_list[:limit]

        items = []
        for result in results_list:
            items.append(
                AttackResultQueryResponse(
                    id=result.conversation_id,  # Use conversation_id as identifier
                    conversation_id=result.conversation_id,
                    objective=result.objective,
                    attack_identifier=filter_sensitive_fields(result.attack_identifier or {}),
                    outcome=str(result.outcome.value) if result.outcome else None,
                    outcome_reason=result.outcome_reason,
                    executed_turns=result.executed_turns,
                    execution_time_ms=result.execution_time_ms,
                    timestamp=None,  # AttackResult doesn't have timestamp
                )
            )

        # No cursor-based pagination available without timestamps
        return PaginatedResponse(
            items=items,
            pagination=PaginationInfo(limit=limit, has_more=has_more, next_cursor=None, prev_cursor=None),
        )

    async def get_scenario_results(
        self,
        *,
        scenario_name: Optional[str] = None,
        run_state: Optional[str] = None,
        labels: Optional[Dict[str, str]] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 50,
        cursor: Optional[str] = None,
    ) -> PaginatedResponse[ScenarioResultQueryResponse]:
        """
        Query scenario results with pagination.

        Returns:
            PaginatedResponse[ScenarioResultQueryResponse]: Paginated list of scenario results.
        """
        results = self._memory.get_scenario_results(
            scenario_name=scenario_name,
            labels=labels,
            added_after=start_time,
            added_before=end_time,
        )

        # Apply run_state filter if provided (not directly supported in API)
        if run_state:
            results = [r for r in results if r.scenario_run_state == run_state]

        # Sort by completion_time descending
        results_list = list(results)
        results_list = sorted(results_list, key=lambda r: r.completion_time or datetime.min, reverse=True)
        has_more = len(results_list) > limit
        results_list = results_list[:limit]

        items = []
        for result in results_list:
            items.append(
                ScenarioResultQueryResponse(
                    id=str(result.id),
                    scenario_name=result.scenario_identifier.name if result.scenario_identifier else "",
                    scenario_description=result.scenario_identifier.description if result.scenario_identifier else "",
                    scenario_version=result.scenario_identifier.version if result.scenario_identifier else 0,
                    pyrit_version=result.scenario_identifier.pyrit_version if result.scenario_identifier else "",
                    run_state=result.scenario_run_state,
                    objective_target_identifier=filter_sensitive_fields(result.objective_target_identifier or {}),
                    labels=result.labels,
                    number_tries=result.number_tries,
                    completion_time=result.completion_time,
                    timestamp=result.completion_time,  # Use completion_time as timestamp
                )
            )

        next_cursor = None
        if has_more and results_list:
            last = results_list[-1]
            next_cursor = _build_cursor(last.completion_time or datetime.min, str(last.id))

        return PaginatedResponse(
            items=items,
            pagination=PaginationInfo(limit=limit, has_more=has_more, next_cursor=next_cursor, prev_cursor=None),
        )

    async def get_seeds(
        self,
        *,
        dataset_name: Optional[str] = None,
        seed_type: Optional[str] = None,
        harm_categories: Optional[List[str]] = None,
        data_type: Optional[str] = None,
        search: Optional[str] = None,
        limit: int = 50,
        cursor: Optional[str] = None,
    ) -> PaginatedResponse[SeedQueryResponse]:
        """
        Query seeds with pagination.

        Returns:
            PaginatedResponse[SeedQueryResponse]: Paginated list of seeds.
        """
        # Build query params - seed_type needs conversion to SeedType
        query_params: Dict[str, Any] = {
            "dataset_name": dataset_name,
            "harm_categories": harm_categories,
        }
        if seed_type:
            query_params["seed_type"] = seed_type
        if data_type:
            query_params["data_types"] = [data_type]
        if search:
            query_params["value"] = search

        seeds = self._memory.get_seeds(**query_params)

        # Sort by date_added descending
        seeds_list = sorted(list(seeds), key=lambda s: s.date_added or datetime.min, reverse=True)
        has_more = len(seeds_list) > limit
        seeds_list = seeds_list[:limit]

        items = []
        for seed in seeds_list:
            # Determine seed_type based on class
            if isinstance(seed, SeedObjective):
                determined_seed_type = "objective"
            elif isinstance(seed, SeedSimulatedConversation):
                determined_seed_type = "simulated_conversation"
            else:
                determined_seed_type = "prompt"

            items.append(
                SeedQueryResponse(
                    id=str(seed.id),
                    value=seed.value,
                    data_type=seed.data_type,
                    name=seed.name,
                    dataset_name=seed.dataset_name,
                    seed_type=determined_seed_type,  # type: ignore
                    harm_categories=list(seed.harm_categories) if seed.harm_categories else None,
                    description=seed.description,
                    source=seed.source,
                    date_added=seed.date_added,
                )
            )

        next_cursor = None
        if has_more and seeds:
            last = seeds[-1]
            next_cursor = _build_cursor(last.date_added or datetime.min, str(last.id))

        return PaginatedResponse(
            items=items,
            pagination=PaginationInfo(limit=limit, has_more=has_more, next_cursor=next_cursor, prev_cursor=None),
        )


# Singleton instance
_memory_service: Optional[MemoryService] = None


def get_memory_service() -> MemoryService:
    """
    Get the memory service singleton.

    Returns:
        MemoryService: The memory service instance.
    """
    global _memory_service
    if _memory_service is None:
        _memory_service = MemoryService()
    return _memory_service
