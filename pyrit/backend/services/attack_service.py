# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
Attack service for managing attacks.

All interactions are modeled as "attacks" - this is the attack-centric API design.
Handles attack lifecycle, message sending, prepended conversations, and scoring.
"""

import uuid
from collections import defaultdict
from datetime import datetime, timezone
from typing import Dict, List, Literal, Optional, cast

from pydantic import BaseModel

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
    UpdateAttackRequest,
)
from pyrit.backend.models.common import PaginationInfo
from pyrit.backend.services.converter_service import get_converter_service
from pyrit.backend.services.target_service import get_target_service
from pyrit.memory import CentralMemory


class AttackState(BaseModel):
    """Internal state for an active attack."""

    attack_id: str
    name: Optional[str] = None
    target_id: str
    target_type: str
    outcome: Optional[Literal["pending", "success", "failure"]] = None
    prepended_conversation: List[Message] = []
    message_count: int = 0
    created_at: datetime
    updated_at: datetime


class AttackService:
    """Service for managing attacks."""

    def __init__(self) -> None:
        """Initialize the attack service."""
        self._memory = CentralMemory.get_memory_instance()
        # Active attack states
        self._attacks: Dict[str, AttackState] = {}
        # Messages by attack ID (in-memory for now)
        self._messages: Dict[str, List[Message]] = defaultdict(list)

    async def list_attacks(
        self,
        target_id: Optional[str] = None,
        outcome: Optional[Literal["pending", "success", "failure"]] = None,
        limit: int = 20,
        cursor: Optional[str] = None,
    ) -> AttackListResponse:
        """
        List attacks with optional filtering and pagination.

        Args:
            target_id: Filter by target instance ID
            outcome: Filter by outcome
            limit: Maximum items per page
            cursor: Pagination cursor

        Returns:
            AttackListResponse: Paginated list of attack summaries
        """
        attacks = list(self._attacks.values())

        # Apply filters
        if target_id:
            attacks = [a for a in attacks if a.target_id == target_id]
        if outcome:
            attacks = [a for a in attacks if a.outcome == outcome]

        # Sort by updated_at descending
        attacks.sort(key=lambda a: a.updated_at, reverse=True)

        # Simple cursor-based pagination (cursor is the attack_id)
        start_idx = 0
        if cursor:
            for i, attack in enumerate(attacks):
                if attack.attack_id == cursor:
                    start_idx = i + 1
                    break

        page = attacks[start_idx : start_idx + limit]
        has_more = len(attacks) > start_idx + limit

        summaries = []
        for attack in page:
            messages = self._messages.get(attack.attack_id, [])
            last_message_preview = None
            if messages:
                last_msg = messages[-1]
                if last_msg.pieces:
                    preview_text = last_msg.pieces[0].converted_value
                    last_message_preview = preview_text[:100] + "..." if len(preview_text) > 100 else preview_text

            summaries.append(
                AttackSummary(
                    attack_id=attack.attack_id,
                    name=attack.name,
                    target_id=attack.target_id,
                    target_type=attack.target_type,
                    outcome=attack.outcome,
                    last_message_preview=last_message_preview,
                    message_count=len(messages),
                    created_at=attack.created_at,
                    updated_at=attack.updated_at,
                )
            )

        next_cursor = page[-1].attack_id if has_more and page else None

        return AttackListResponse(
            items=summaries,
            pagination=PaginationInfo(
                limit=limit,
                has_more=has_more,
                next_cursor=next_cursor,
                prev_cursor=cursor,
            ),
        )

    async def get_attack(self, attack_id: str) -> Optional[AttackDetail]:
        """
        Get attack details including all messages.

        Args:
            attack_id: Attack ID

        Returns:
            AttackDetail or None if not found
        """
        state = self._attacks.get(attack_id)
        if not state:
            return None

        messages = self._messages.get(attack_id, [])

        return AttackDetail(
            attack_id=state.attack_id,
            name=state.name,
            target_id=state.target_id,
            target_type=state.target_type,
            outcome=state.outcome,
            prepended_conversation=state.prepended_conversation,
            messages=messages,
            created_at=state.created_at,
            updated_at=state.updated_at,
        )

    async def create_attack(
        self,
        request: CreateAttackRequest,
    ) -> CreateAttackResponse:
        """
        Create a new attack.

        Args:
            request: Attack creation request

        Returns:
            CreateAttackResponse: Created attack details
        """
        target_service = get_target_service()

        # Validate target exists
        target_instance = await target_service.get_target(request.target_id)
        if not target_instance:
            raise ValueError(f"Target instance '{request.target_id}' not found")

        attack_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc)

        # Convert prepended messages to Message format
        prepended_messages: List[Message] = []
        if request.prepended_conversation:
            for i, prep_msg in enumerate(request.prepended_conversation):
                msg = Message(
                    message_id=str(uuid.uuid4()),
                    turn_number=0,  # Prepended messages are turn 0
                    role=prep_msg.role,
                    pieces=[
                        MessagePiece(
                            piece_id=str(uuid.uuid4()),
                            data_type="text",
                            original_value=prep_msg.content,
                            converted_value=prep_msg.content,
                            scores=[],
                        )
                    ],
                    created_at=now,
                )
                prepended_messages.append(msg)

        state = AttackState(
            attack_id=attack_id,
            name=request.name,
            target_id=request.target_id,
            target_type=target_instance.type,
            outcome=None,
            prepended_conversation=prepended_messages,
            message_count=0,
            created_at=now,
            updated_at=now,
        )
        self._attacks[attack_id] = state

        return CreateAttackResponse(
            attack_id=attack_id,
            created_at=now,
        )

    async def update_attack(
        self,
        attack_id: str,
        request: UpdateAttackRequest,
    ) -> Optional[AttackDetail]:
        """
        Update an attack's outcome.

        Args:
            attack_id: Attack ID
            request: Update request with outcome

        Returns:
            Updated AttackDetail or None if not found
        """
        state = self._attacks.get(attack_id)
        if not state:
            return None

        state.outcome = request.outcome
        state.updated_at = datetime.now(timezone.utc)

        return await self.get_attack(attack_id)

    async def add_message(
        self,
        attack_id: str,
        request: AddMessageRequest,
    ) -> AddMessageResponse:
        """
        Add a message to an attack.

        If send=True, sends to target and waits for response.
        If send=False, just stores the message in memory.

        Args:
            attack_id: Attack ID
            request: Add message request

        Returns:
            AddMessageResponse: Updated attack detail
        """
        state = self._attacks.get(attack_id)
        if not state:
            raise ValueError(f"Attack '{attack_id}' not found")

        target_service = get_target_service()
        converter_service = get_converter_service()

        now = datetime.now(timezone.utc)
        state.message_count += 1
        msg_turn = state.message_count

        # Determine which converters to use (only for user messages being sent)
        converters = []
        if request.send and request.role == "user" and request.converter_ids:
            converters = converter_service.get_converter_objects_for_ids(request.converter_ids)

        # Build message pieces
        msg_pieces: List[MessagePiece] = []
        for piece_req in request.pieces:
            original_value = piece_req.content
            converted_value = original_value

            # Apply converters
            for converter in converters:
                result = await converter.convert_async(prompt=converted_value)
                converted_value = result.output_text

            msg_pieces.append(
                MessagePiece(
                    piece_id=str(uuid.uuid4()),
                    data_type=piece_req.data_type,
                    original_value=original_value,
                    original_value_mime_type=piece_req.mime_type,
                    converted_value=converted_value,
                    converted_value_mime_type=piece_req.mime_type,
                    scores=[],
                )
            )

        message = Message(
            message_id=str(uuid.uuid4()),
            turn_number=msg_turn,
            role=request.role,
            pieces=msg_pieces,
            created_at=now,
        )

        # Store the message
        self._messages[attack_id].append(message)

        # If send=True, send to target and get response
        transport_error: Optional[str] = None
        if request.send:
            target_obj = target_service.get_target_object(state.target_id)
            if not target_obj:
                raise ValueError(f"Target object for '{state.target_id}' not found")

            try:
                # Build conversation for target
                from pyrit.models import Message as PyritMessage
                from pyrit.models import MessagePiece as PyritMessagePiece
                from pyrit.models import PromptDataType

                # Create prompt pieces for target
                prompt_pieces = []
                for piece in msg_pieces:
                    pyrit_piece = PyritMessagePiece(
                        role=request.role,
                        original_value=piece.original_value or "",
                        original_value_data_type=cast(PromptDataType, piece.data_type),
                        converted_value=piece.converted_value,
                        converted_value_data_type=cast(PromptDataType, piece.data_type),
                        conversation_id=attack_id,
                        sequence=msg_turn,
                    )
                    prompt_pieces.append(pyrit_piece)

                pyrit_message = PyritMessage(prompt_pieces)

                # Send to target
                response_messages = await target_obj.send_prompt_async(message=pyrit_message)

                # Build assistant response
                state.message_count += 1
                assistant_turn = state.message_count

                assistant_pieces: List[MessagePiece] = []
                if response_messages:
                    for resp_msg in response_messages:
                        for resp_piece in resp_msg.message_pieces:
                            assistant_pieces.append(
                                MessagePiece(
                                    piece_id=str(uuid.uuid4()),
                                    data_type=resp_piece.converted_value_data_type or "text",
                                    original_value=resp_piece.original_value,
                                    converted_value=resp_piece.converted_value or "",
                                    scores=[],
                                    response_error=resp_piece.response_error,
                                )
                            )

                assistant_message = Message(
                    message_id=str(uuid.uuid4()),
                    turn_number=assistant_turn,
                    role="assistant",
                    pieces=assistant_pieces
                    if assistant_pieces
                    else [
                        MessagePiece(
                            piece_id=str(uuid.uuid4()),
                            data_type="text",
                            converted_value="",
                            scores=[],
                        )
                    ],
                    created_at=datetime.now(timezone.utc),
                )

                # Store assistant message
                self._messages[attack_id].append(assistant_message)

            except Exception as e:
                transport_error = str(e)

        # Update attack timestamp
        state.updated_at = datetime.now(timezone.utc)

        # Get updated attack detail
        attack_detail = await self.get_attack(attack_id)
        if attack_detail is None:
            raise ValueError(f"Attack '{attack_id}' not found after update")

        return AddMessageResponse(
            attack=attack_detail,
            error=transport_error,
        )

    async def delete_attack(self, attack_id: str) -> bool:
        """
        Delete an attack.

        Args:
            attack_id: Attack ID

        Returns:
            True if deleted, False if not found
        """
        if attack_id in self._attacks:
            del self._attacks[attack_id]
            self._messages.pop(attack_id, None)
            return True
        return False


# Global service instance
_attack_service: Optional[AttackService] = None


def get_attack_service() -> AttackService:
    """
    Get the global attack service instance.

    Returns:
        AttackService: The singleton attack service instance.
    """
    global _attack_service
    if _attack_service is None:
        _attack_service = AttackService()
    return _attack_service
