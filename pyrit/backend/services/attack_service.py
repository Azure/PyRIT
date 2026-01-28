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
- For GUI use, it's a 1-to-1 mapping: one AttackResult, one conversation
- Future AI-generated attacks may have multiple related conversations
"""

import uuid
from datetime import datetime, timezone
from typing import List, Literal, Optional, cast

from pyrit.backend.models.attacks import (
    AddMessageRequest,
    AddMessageResponse,
    AttackDetail,
    AttackListResponse,
    AttackSummary,
    CreateAttackRequest,
    CreateAttackResponse,
    Message,
    MessagePiece,
    Score,
    UpdateAttackRequest,
)
from pyrit.backend.models.common import PaginationInfo
from pyrit.backend.services.converter_service import get_converter_service
from pyrit.backend.services.target_service import get_target_service
from pyrit.memory import CentralMemory
from pyrit.models import AttackOutcome, AttackResult
from pyrit.models import Message as PyritMessage
from pyrit.models import MessagePiece as PyritMessagePiece
from pyrit.models import PromptDataType
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

    async def list_attacks(
        self,
        *,
        target_id: Optional[str] = None,
        outcome: Optional[Literal["pending", "success", "failure"]] = None,
        limit: int = 20,
        cursor: Optional[str] = None,
    ) -> AttackListResponse:
        """
        List attacks with optional filtering and pagination.

        Queries AttackResult entries from the database.
        """
        # Map outcome string to AttackOutcome enum value for filtering
        outcome_filter = None
        if outcome:
            outcome_map = {
                "pending": "undetermined",
                "success": "success",
                "failure": "failure",
            }
            outcome_filter = outcome_map.get(outcome)

        attack_results = self._memory.get_attack_results(outcome=outcome_filter)

        # Convert to summaries and filter
        summaries = []
        for ar in attack_results:
            # Get target info from attack_identifier
            ar_target_id = ar.attack_identifier.get("target_id", "")
            if target_id and ar_target_id != target_id:
                continue

            summary = self._build_summary(ar)
            summaries.append(summary)

        # Sort by most recent
        summaries.sort(key=lambda s: s.updated_at, reverse=True)

        # Paginate
        page, has_more = self._paginate(summaries, cursor, limit)
        next_cursor = page[-1].attack_id if has_more and page else None

        return AttackListResponse(
            items=page,
            pagination=PaginationInfo(
                limit=limit, has_more=has_more, next_cursor=next_cursor, prev_cursor=cursor
            ),
        )

    async def get_attack(self, attack_id: str) -> Optional[AttackDetail]:
        """
        Get attack details including all messages.

        Queries the AttackResult and its conversation from the database.
        """
        # Get the attack result
        results = self._memory.get_attack_results(conversation_id=attack_id)
        if not results:
            return None

        ar = results[0]

        # Get messages for this conversation
        pyrit_messages = self._memory.get_conversation(conversation_id=attack_id)
        backend_messages = self._convert_pyrit_messages(pyrit_messages)

        return AttackDetail(
            attack_id=attack_id,
            name=ar.attack_identifier.get("name"),
            target_id=ar.attack_identifier.get("target_id", ""),
            target_type=ar.attack_identifier.get("target_type", ""),
            outcome=self._map_outcome(ar.outcome),
            prepended_conversation=[],
            messages=backend_messages,
            created_at=ar.metadata.get("created_at", datetime.now(timezone.utc)),
            updated_at=ar.metadata.get("updated_at", datetime.now(timezone.utc)),
        )

    async def create_attack(self, request: CreateAttackRequest) -> CreateAttackResponse:
        """
        Create a new attack.

        Creates an AttackResult with a new conversation_id.
        """
        target_service = get_target_service()
        target_instance = await target_service.get_target(request.target_id)
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
                **(request.labels or {}),
            },
        )

        # Store in memory
        self._memory.add_attack_results_to_memory(attack_results=[attack_result])

        # Store prepended conversation if provided
        if request.prepended_conversation:
            await self._store_prepended_messages(
                conversation_id=conversation_id,
                prepended=request.prepended_conversation,
            )

        return CreateAttackResponse(attack_id=conversation_id, created_at=now)

    async def update_attack(
        self, attack_id: str, request: UpdateAttackRequest
    ) -> Optional[AttackDetail]:
        """
        Update an attack's outcome.

        Updates the AttackResult in the database.
        """
        results = self._memory.get_attack_results(conversation_id=attack_id)
        if not results:
            return None

        # Map outcome
        outcome_map = {
            "pending": AttackOutcome.UNDETERMINED,
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

        return await self.get_attack(attack_id)

    async def add_message(
        self, attack_id: str, request: AddMessageRequest
    ) -> AddMessageResponse:
        """
        Add a message to an attack, optionally sending to target.

        Messages are stored in the database via PromptNormalizer.
        """
        # Check if attack exists
        results = self._memory.get_attack_results(conversation_id=attack_id)
        if not results:
            raise ValueError(f"Attack '{attack_id}' not found")

        ar = results[0]
        target_id = ar.attack_identifier.get("target_id")
        if not target_id:
            raise ValueError(f"Attack '{attack_id}' has no target configured")

        # Get existing messages to determine sequence
        existing = self._memory.get_message_pieces(conversation_id=attack_id)
        sequence = max((p.sequence for p in existing), default=-1) + 1

        if request.send:
            await self._send_and_store_message(attack_id, target_id, request, sequence)
        else:
            await self._store_message_only(attack_id, request, sequence)

        # Update attack timestamp
        ar.metadata["updated_at"] = datetime.now(timezone.utc).isoformat()

        attack_detail = await self.get_attack(attack_id)
        if attack_detail is None:
            raise ValueError(f"Attack '{attack_id}' not found after update")

        return AddMessageResponse(attack=attack_detail)

    # ========================================================================
    # Private Helper Methods - Summary Building
    # ========================================================================

    def _build_summary(self, ar: AttackResult) -> AttackSummary:
        """Build an AttackSummary from an AttackResult."""
        # Get message count and last preview
        pieces = self._memory.get_message_pieces(conversation_id=ar.conversation_id)
        message_count = len(set(p.sequence for p in pieces))
        last_preview = None
        if pieces:
            last_piece = max(pieces, key=lambda p: p.sequence)
            text = last_piece.converted_value or ""
            last_preview = text[:100] + "..." if len(text) > 100 else text

        created_str = ar.metadata.get("created_at")
        updated_str = ar.metadata.get("updated_at")
        created_at = datetime.fromisoformat(created_str) if created_str else datetime.now(timezone.utc)
        updated_at = datetime.fromisoformat(updated_str) if updated_str else created_at

        return AttackSummary(
            attack_id=ar.conversation_id,
            name=ar.attack_identifier.get("name"),
            target_id=ar.attack_identifier.get("target_id", ""),
            target_type=ar.attack_identifier.get("target_type", ""),
            outcome=self._map_outcome(ar.outcome),
            last_message_preview=last_preview,
            message_count=message_count,
            created_at=created_at,
            updated_at=updated_at,
        )

    def _map_outcome(self, outcome: AttackOutcome) -> Optional[Literal["pending", "success", "failure"]]:
        """Map AttackOutcome enum to API outcome string."""
        if outcome == AttackOutcome.SUCCESS:
            return "success"
        elif outcome == AttackOutcome.FAILURE:
            return "failure"
        else:
            return "pending"

    # ========================================================================
    # Private Helper Methods - Pagination
    # ========================================================================

    def _paginate(
        self, items: List[AttackSummary], cursor: Optional[str], limit: int
    ) -> tuple[List[AttackSummary], bool]:
        """Apply cursor-based pagination."""
        start_idx = 0
        if cursor:
            for i, item in enumerate(items):
                if item.attack_id == cursor:
                    start_idx = i + 1
                    break

        page = items[start_idx : start_idx + limit]
        has_more = len(items) > start_idx + limit
        return page, has_more

    # ========================================================================
    # Private Helper Methods - Message Conversion
    # ========================================================================

    def _convert_pyrit_messages(self, pyrit_messages: list) -> List[Message]:
        """Convert PyRIT messages to backend Message format."""
        messages = []
        for msg in pyrit_messages:
            pieces = [
                MessagePiece(
                    piece_id=str(p.id),
                    data_type=p.converted_value_data_type or "text",
                    original_value=p.original_value,
                    converted_value=p.converted_value or "",
                    scores=self._convert_scores(p.scores) if hasattr(p, 'scores') and p.scores else [],
                    response_error=p.response_error or "none",
                )
                for p in msg.message_pieces
            ]

            first = msg.message_pieces[0] if msg.message_pieces else None
            messages.append(Message(
                message_id=str(first.id) if first else str(uuid.uuid4()),
                turn_number=first.sequence if first else 0,
                role=first.role if first else "user",
                pieces=pieces,
                created_at=first.timestamp if first else datetime.now(timezone.utc),
            ))

        return messages

    def _convert_scores(self, scores: list) -> List[Score]:
        """Convert PyRIT scores to backend Score format."""
        return [
            Score(
                score_id=str(s.id),
                scorer_type=s.scorer_class_identifier.get("__type__", "unknown"),
                score_value=s.score_value,
                score_rationale=s.score_rationale,
                scored_at=s.timestamp,
            )
            for s in scores
        ]

    # ========================================================================
    # Private Helper Methods - Store Messages
    # ========================================================================

    async def _store_prepended_messages(
        self,
        conversation_id: str,
        prepended: list,
    ) -> None:
        """Store prepended conversation messages in memory."""
        for seq, msg in enumerate(prepended):
            piece = PyritMessagePiece(
                role=msg.role,
                original_value=msg.content,
                original_value_data_type="text",
                converted_value=msg.content,
                converted_value_data_type="text",
                conversation_id=conversation_id,
                sequence=seq,
            )
            self._memory.add_message_pieces_to_memory(message_pieces=[piece])

    async def _send_and_store_message(
        self,
        attack_id: str,
        target_id: str,
        request: AddMessageRequest,
        sequence: int,
    ) -> None:
        """Send message to target via normalizer and store response."""
        target_obj = get_target_service().get_target_object(target_id)
        if not target_obj:
            raise ValueError(f"Target object for '{target_id}' not found")

        pyrit_message = self._build_pyrit_message(request, attack_id, sequence)
        converter_configs = self._get_converter_configs(request)

        normalizer = PromptNormalizer()
        await normalizer.send_prompt_async(
            message=pyrit_message,
            target=target_obj,
            conversation_id=attack_id,
            request_converter_configurations=converter_configs,
        )
        # PromptNormalizer stores both request and response in memory automatically

    async def _store_message_only(
        self,
        attack_id: str,
        request: AddMessageRequest,
        sequence: int,
    ) -> None:
        """Store message without sending (send=False)."""
        for p in request.pieces:
            piece = PyritMessagePiece(
                role=request.role,
                original_value=p.original_value,
                original_value_data_type=cast(PromptDataType, p.data_type),
                converted_value=p.converted_value or p.original_value,
                converted_value_data_type=cast(PromptDataType, p.data_type),
                conversation_id=attack_id,
                sequence=sequence,
            )
            self._memory.add_message_pieces_to_memory(message_pieces=[piece])

    def _build_pyrit_message(
        self,
        request: AddMessageRequest,
        conversation_id: str,
        sequence: int,
    ) -> PyritMessage:
        """Build PyRIT Message from request."""
        pieces = [
            PyritMessagePiece(
                role=request.role,
                original_value=p.original_value,
                original_value_data_type=cast(PromptDataType, p.data_type),
                converted_value=p.converted_value or p.original_value,
                converted_value_data_type=cast(PromptDataType, p.data_type),
                conversation_id=conversation_id,
                sequence=sequence,
            )
            for p in request.pieces
        ]
        return PyritMessage(pieces)

    def _get_converter_configs(
        self, request: AddMessageRequest
    ) -> List[PromptConverterConfiguration]:
        """Get converter configurations if needed."""
        has_preconverted = any(p.converted_value is not None for p in request.pieces)
        if has_preconverted or not request.converter_ids:
            return []

        converters = get_converter_service().get_converter_objects_for_ids(request.converter_ids)
        return PromptConverterConfiguration.from_converters(converters=converters)


# ============================================================================
# Singleton
# ============================================================================

_attack_service: Optional[AttackService] = None


def get_attack_service() -> AttackService:
    """
    Get the global attack service instance.

    Returns:
        The singleton AttackService instance.
    """
    global _attack_service
    if _attack_service is None:
        _attack_service = AttackService()
    return _attack_service
