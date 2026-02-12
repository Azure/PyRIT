# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
Attack service for managing attacks.

All user interactions are modeled as "attacks" - this is the attack-centric API design.
Handles attack lifecycle, message sending, and scoring.

ARCHITECTURE:
- Each attack is represented by an AttackResult stored in the database
- The AttackResult has a conversation_id that links to the main conversation
- Messages are stored via PyRIT memory with that conversation_id
- For human-led attacks, it's a 1-to-1 mapping: one AttackResult, one conversation
- AI-generated attacks may have multiple related conversations
"""

import uuid
from datetime import datetime, timezone
from functools import lru_cache
from typing import Any, Dict, List, Literal, Optional

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
from pyrit.backend.models.common import PaginationInfo
from pyrit.backend.mappers.attack_mappers import (
    attack_result_to_summary,
    pyrit_messages_to_dto,
    request_piece_to_pyrit_message_piece,
    request_to_pyrit_message,
)
from pyrit.backend.services.converter_service import get_converter_service
from pyrit.backend.services.target_service import get_target_service
from pyrit.memory import CentralMemory
from pyrit.models import AttackOutcome, AttackResult
from pyrit.prompt_normalizer import PromptConverterConfiguration, PromptNormalizer


class AttackService:
    """
    Service for managing attacks.

    Uses PyRIT memory (database) as the source of truth via AttackResult.
    """

    def __init__(self) -> None:
        """Initialize the attack service."""
        self._memory = CentralMemory.get_memory_instance()

    # ========================================================================
    # Public API Methods
    # ========================================================================

    async def list_attacks_async(
        self,
        *,
        target_id: Optional[str] = None,
        outcome: Optional[Literal["undetermined", "success", "failure"]] = None,
        name: Optional[str] = None,
        labels: Optional[Dict[str, str]] = None,
        min_turns: Optional[int] = None,
        max_turns: Optional[int] = None,
        limit: int = 20,
        cursor: Optional[str] = None,
    ) -> AttackListResponse:
        """
        List attacks with optional filtering and pagination.

        Queries AttackResult entries from the database.

        Args:
            target_id: Filter by target instance ID (from attack_identifier).
            outcome: Filter by attack outcome.
            name: Filter by attack name (substring match on attack_identifier.name).
            labels: Filter by labels (all must match).
            min_turns: Filter by minimum executed turns.
            max_turns: Filter by maximum executed turns.
            limit: Maximum items to return.
            cursor: Pagination cursor.

        Returns:
            AttackListResponse with filtered and paginated attack summaries.
        """
        # Phase 1: Query + lightweight filtering (no pieces needed)
        attack_results = self._memory.get_attack_results(
            outcome=outcome,
            labels=labels,
        )

        filtered: List[AttackResult] = []
        for ar in attack_results:
            if target_id and ar.attack_identifier.get("target_id", "") != target_id:
                continue
            if name and name.lower() not in ar.attack_identifier.get("name", "").lower():
                continue
            if min_turns is not None and ar.executed_turns < min_turns:
                continue
            if max_turns is not None and ar.executed_turns > max_turns:
                continue
            filtered.append(ar)

        # Sort by most recent (metadata lives on AttackResult, no pieces needed)
        filtered.sort(
            key=lambda ar: ar.metadata.get("updated_at", ar.metadata.get("created_at", "")),
            reverse=True,
        )

        # Paginate on the lightweight list first
        page_results, has_more = self._paginate_attack_results(filtered, cursor, limit)
        next_cursor = page_results[-1].conversation_id if has_more and page_results else None

        # Phase 2: Fetch pieces only for the page we're returning
        page: List[AttackSummary] = []
        for ar in page_results:
            pieces = self._memory.get_message_pieces(conversation_id=ar.conversation_id)
            page.append(attack_result_to_summary(ar, pieces=pieces))

        return AttackListResponse(
            items=page,
            pagination=PaginationInfo(limit=limit, has_more=has_more, next_cursor=next_cursor, prev_cursor=cursor),
        )

    async def get_attack_async(self, *, attack_id: str) -> Optional[AttackSummary]:
        """
        Get attack details (high-level metadata, no messages).

        Queries the AttackResult from the database.

        Returns:
            AttackSummary if found, None otherwise.
        """
        results = self._memory.get_attack_results(conversation_id=attack_id)
        if not results:
            return None

        ar = results[0]
        pieces = self._memory.get_message_pieces(conversation_id=ar.conversation_id)
        return attack_result_to_summary(ar, pieces=pieces)

    async def get_attack_messages_async(self, *, attack_id: str) -> Optional[AttackMessagesResponse]:
        """
        Get all messages for an attack.

        Returns:
            AttackMessagesResponse if attack found, None otherwise.
        """
        # Check attack exists
        results = self._memory.get_attack_results(conversation_id=attack_id)
        if not results:
            return None

        # Get messages for this conversation
        pyrit_messages = self._memory.get_conversation(conversation_id=attack_id)
        backend_messages = pyrit_messages_to_dto(list(pyrit_messages))

        return AttackMessagesResponse(
            attack_id=attack_id,
            messages=backend_messages,
        )

    async def create_attack_async(self, *, request: CreateAttackRequest) -> CreateAttackResponse:
        """
        Create a new attack.

        Creates an AttackResult with a new conversation_id.

        Returns:
            CreateAttackResponse with the new attack's ID and creation time.
        """
        target_service = get_target_service()
        target_instance = await target_service.get_target_async(target_id=request.target_id)
        if not target_instance:
            raise ValueError(f"Target instance '{request.target_id}' not found")

        # Generate conversation_id (this is the attack_id)
        conversation_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc)

        # Create AttackResult
        attack_result = AttackResult(
            conversation_id=conversation_id,
            objective=request.name or "Manual attack via GUI",
            attack_identifier={
                "name": request.name or "",
                "target_id": request.target_id,
                "target_type": target_instance.type,
                "source": "gui",
            },
            outcome=AttackOutcome.UNDETERMINED,
            metadata={
                "created_at": now.isoformat(),
                "updated_at": now.isoformat(),
            },
        )

        # Store in memory
        self._memory.add_attack_results_to_memory(attack_results=[attack_result])

        # Store prepended conversation if provided
        if request.prepended_conversation:
            await self._store_prepended_messages(
                conversation_id=conversation_id,
                prepended=request.prepended_conversation,
                labels=request.labels,
            )

        return CreateAttackResponse(attack_id=conversation_id, created_at=now)

    async def update_attack_async(self, *, attack_id: str, request: UpdateAttackRequest) -> Optional[AttackSummary]:
        """
        Update an attack's outcome.

        Updates the AttackResult in the database.

        Returns:
            Updated AttackSummary if found, None otherwise.
        """
        results = self._memory.get_attack_results(conversation_id=attack_id)
        if not results:
            return None

        # Map outcome
        outcome_map = {
            "undetermined": AttackOutcome.UNDETERMINED,
            "success": AttackOutcome.SUCCESS,
            "failure": AttackOutcome.FAILURE,
        }
        new_outcome = outcome_map.get(request.outcome, AttackOutcome.UNDETERMINED)

        # Update the attack result (need to update via memory interface)
        # For now, we update metadata to track the change
        ar = results[0]
        ar.outcome = new_outcome
        ar.metadata["updated_at"] = datetime.now(timezone.utc).isoformat()

        # Re-add to memory (this should update)
        self._memory.add_attack_results_to_memory(attack_results=[ar])

        return await self.get_attack_async(attack_id=attack_id)

    async def add_message_async(self, *, attack_id: str, request: AddMessageRequest) -> AddMessageResponse:
        """
        Add a message to an attack, optionally sending to target.

        Messages are stored in the database via PromptNormalizer.

        Returns:
            AddMessageResponse containing the updated attack detail.
        """
        # Check if attack exists
        results = self._memory.get_attack_results(conversation_id=attack_id)
        if not results:
            raise ValueError(f"Attack '{attack_id}' not found")

        ar = results[0]
        target_id = ar.attack_identifier.get("target_id")
        if not target_id:
            raise ValueError(f"Attack '{attack_id}' has no target configured")

        # Get existing messages to determine sequence.
        # NOTE: This read-then-write is not atomic (TOCTOU). Fine for the
        # current single-user UI, but would need a DB-level sequence
        # generator or optimistic locking if concurrent writes are supported.
        existing = self._memory.get_message_pieces(conversation_id=attack_id)
        sequence = max((p.sequence for p in existing), default=-1) + 1

        # Inherit labels from existing pieces so new messages stay consistent
        attack_labels = next((p.labels for p in existing if getattr(p, "labels", None)), None)

        if request.send:
            await self._send_and_store_message(attack_id, target_id, request, sequence, labels=attack_labels)
        else:
            await self._store_message_only(attack_id, request, sequence, labels=attack_labels)

        # Update attack timestamp
        ar.metadata["updated_at"] = datetime.now(timezone.utc).isoformat()

        attack_detail = await self.get_attack_async(attack_id=attack_id)
        if attack_detail is None:
            raise ValueError(f"Attack '{attack_id}' not found after update")

        attack_messages = await self.get_attack_messages_async(attack_id=attack_id)
        if attack_messages is None:
            raise ValueError(f"Attack '{attack_id}' messages not found after update")

        return AddMessageResponse(attack=attack_detail, messages=attack_messages)

    # ========================================================================
    # Private Helper Methods - Pagination
    # ========================================================================

    def _paginate_attack_results(
        self, items: List[AttackResult], cursor: Optional[str], limit: int
    ) -> tuple[List[AttackResult], bool]:
        """
        Apply cursor-based pagination over AttackResult objects.

        Operates on lightweight AttackResult objects before pieces are fetched,
        so only the final page incurs per-attack piece queries.

        Returns:
            Tuple of (paginated items, has_more flag).
        """
        start_idx = 0
        if cursor:
            for i, item in enumerate(items):
                if item.conversation_id == cursor:
                    start_idx = i + 1
                    break

        page = items[start_idx : start_idx + limit]
        has_more = len(items) > start_idx + limit
        return page, has_more

    # ========================================================================
    # Private Helper Methods - Store Messages
    # ========================================================================

    async def _store_prepended_messages(
        self,
        conversation_id: str,
        prepended: List[Any],
        labels: Optional[Dict[str, str]] = None,
    ) -> None:
        """Store prepended conversation messages in memory."""
        seq = 0
        for msg in prepended:
            for p in msg.pieces:
                piece = request_piece_to_pyrit_message_piece(
                    piece=p,
                    role=msg.role,
                    conversation_id=conversation_id,
                    sequence=seq,
                    labels=labels,
                )
                self._memory.add_message_pieces_to_memory(message_pieces=[piece])
            seq += 1

    async def _send_and_store_message(
        self,
        attack_id: str,
        target_id: str,
        request: AddMessageRequest,
        sequence: int,
        *,
        labels: Optional[Dict[str, str]] = None,
    ) -> None:
        """Send message to target via normalizer and store response."""
        target_obj = get_target_service().get_target_object(target_id=target_id)
        if not target_obj:
            raise ValueError(f"Target object for '{target_id}' not found")

        pyrit_message = request_to_pyrit_message(
            request=request,
            conversation_id=attack_id,
            sequence=sequence,
            labels=labels,
        )
        converter_configs = self._get_converter_configs(request)

        normalizer = PromptNormalizer()
        await normalizer.send_prompt_async(
            message=pyrit_message,
            target=target_obj,
            conversation_id=attack_id,
            request_converter_configurations=converter_configs,
            labels=labels,
        )
        # PromptNormalizer stores both request and response in memory automatically

    async def _store_message_only(
        self,
        attack_id: str,
        request: AddMessageRequest,
        sequence: int,
        *,
        labels: Optional[Dict[str, str]] = None,
    ) -> None:
        """Store message without sending (send=False)."""
        for p in request.pieces:
            piece = request_piece_to_pyrit_message_piece(
                piece=p,
                role=request.role,
                conversation_id=attack_id,
                sequence=sequence,
                labels=labels,
            )
            self._memory.add_message_pieces_to_memory(message_pieces=[piece])

    def _get_converter_configs(self, request: AddMessageRequest) -> List[PromptConverterConfiguration]:
        """
        Get converter configurations if needed.

        Returns:
            List of PromptConverterConfiguration for the converters.
        """
        has_preconverted = any(p.converted_value is not None for p in request.pieces)
        if has_preconverted or not request.converter_ids:
            return []

        converters = get_converter_service().get_converter_objects_for_ids(converter_ids=request.converter_ids)
        return PromptConverterConfiguration.from_converters(converters=converters)


# ============================================================================
# Singleton
# ============================================================================


@lru_cache(maxsize=1)
def get_attack_service() -> AttackService:
    """
    Get the global attack service instance.

    Returns:
        The singleton AttackService instance.
    """
    return AttackService()
