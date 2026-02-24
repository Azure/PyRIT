# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

"""
Attack mappers – domain ↔ DTO translation for attack-related models.

All functions are pure (no database or service calls) so they are easy to test.
The one exception is `attack_result_to_summary` which receives pre-fetched pieces.
"""

import mimetypes
import uuid
from datetime import datetime, timezone
from typing import Dict, List, Optional, Sequence, cast

from pyrit.backend.models.attacks import (
    AddMessageRequest,
    AttackSummary,
    Message,
    MessagePiece,
    MessagePieceRequest,
    Score,
)
from pyrit.models import AttackResult, ChatMessageRole, PromptDataType
from pyrit.models import Message as PyritMessage
from pyrit.models import MessagePiece as PyritMessagePiece
from pyrit.models import Score as PyritScore

# ============================================================================
# Domain → DTO  (for API responses)
# ============================================================================


def attack_result_to_summary(
    ar: AttackResult,
    *,
    pieces: Sequence[PyritMessagePiece],
) -> AttackSummary:
    """
    Build an AttackSummary DTO from an AttackResult and its message pieces.

    Extracts only the frontend-relevant fields from the internal identifiers,
    avoiding leakage of internal PyRIT core structures.

    Args:
        ar: The domain AttackResult.
        pieces: Pre-fetched message pieces for this conversation.

    Returns:
        AttackSummary DTO ready for the API response.
    """
    message_count = len({p.sequence for p in pieces})
    last_preview = _get_preview_from_pieces(pieces)

    created_str = ar.metadata.get("created_at")
    updated_str = ar.metadata.get("updated_at")
    created_at = datetime.fromisoformat(created_str) if created_str else datetime.now(timezone.utc)
    updated_at = datetime.fromisoformat(updated_str) if updated_str else created_at

    aid = ar.attack_identifier

    # Extract only frontend-relevant fields from ComponentIdentifier
    target_id = aid.get_child("objective_target") if aid else None
    converter_ids = aid.get_child_list("request_converters") if aid else []

    return AttackSummary(
        conversation_id=ar.conversation_id,
        attack_type=aid.class_name if aid else "Unknown",
        attack_specific_params=aid.params or None if aid else None,
        target_unique_name=target_id.unique_name if target_id else None,
        target_type=target_id.class_name if target_id else None,
        converters=[c.class_name for c in converter_ids] if converter_ids else [],
        outcome=ar.outcome.value,
        last_message_preview=last_preview,
        message_count=message_count,
        labels=_collect_labels_from_pieces(pieces),
        created_at=created_at,
        updated_at=updated_at,
    )


def pyrit_scores_to_dto(scores: List[PyritScore]) -> List[Score]:
    """
    Translate PyRIT score objects to backend Score DTOs.

    Returns:
        List of Score DTOs for the API.
    """
    return [
        Score(
            score_id=str(s.id),
            scorer_type=s.scorer_class_identifier.class_name,
            score_value=float(s.score_value),
            score_rationale=s.score_rationale,
            scored_at=s.timestamp,
        )
        for s in scores
    ]


def _infer_mime_type(*, value: Optional[str], data_type: PromptDataType) -> Optional[str]:
    """
    Infer MIME type from a value and its data type.

    For non-text data types, attempts to guess the MIME type from the value
    treated as a file path (using the file extension).  Returns ``None`` for
    text content or when the type cannot be determined.

    Args:
        value: The value (typically a file path for media content).
        data_type: The prompt data type (e.g., 'text', 'image', 'audio').

    Returns:
        MIME type string (e.g., 'image/png') or None.
    """
    if not value or data_type == "text":
        return None
    mime_type, _ = mimetypes.guess_type(value)
    return mime_type


def pyrit_messages_to_dto(pyrit_messages: List[PyritMessage]) -> List[Message]:
    """
    Translate PyRIT messages to backend Message DTOs.

    Returns:
        List of Message DTOs for the API.
    """
    messages = []
    for msg in pyrit_messages:
        pieces = [
            MessagePiece(
                piece_id=str(p.id),
                original_value_data_type=p.original_value_data_type or "text",
                converted_value_data_type=p.converted_value_data_type or "text",
                original_value=p.original_value,
                original_value_mime_type=_infer_mime_type(
                    value=p.original_value, data_type=p.original_value_data_type or "text"
                ),
                converted_value=p.converted_value or "",
                converted_value_mime_type=_infer_mime_type(
                    value=p.converted_value, data_type=p.converted_value_data_type or "text"
                ),
                scores=pyrit_scores_to_dto(p.scores) if p.scores else [],
                response_error=p.response_error or "none",
            )
            for p in msg.message_pieces
        ]

        first = msg.message_pieces[0] if msg.message_pieces else None
        messages.append(
            Message(
                turn_number=first.sequence if first else 0,
                role=first.get_role_for_storage() if first else "user",
                pieces=pieces,
                created_at=first.timestamp if first else datetime.now(timezone.utc),
            )
        )

    return messages


# ============================================================================
# DTO → Domain  (for inbound requests)
# ============================================================================


def request_piece_to_pyrit_message_piece(
    *,
    piece: MessagePieceRequest,
    role: ChatMessageRole,
    conversation_id: str,
    sequence: int,
    labels: Optional[Dict[str, str]] = None,
) -> PyritMessagePiece:
    """
    Convert a single request piece DTO to a PyRIT MessagePiece domain object.

    Args:
        piece: The request piece (with data_type, original_value, converted_value).
        role: The message role.
        conversation_id: The conversation/attack ID.
        sequence: The message sequence number.
        labels: Optional labels to stamp on the piece.

    Returns:
        PyritMessagePiece domain object.
    """
    metadata: Optional[Dict[str, str | int]] = {"mime_type": piece.mime_type} if piece.mime_type else None
    original_prompt_id = uuid.UUID(piece.original_prompt_id) if piece.original_prompt_id else None
    return PyritMessagePiece(
        role=role,
        original_value=piece.original_value,
        original_value_data_type=cast(PromptDataType, piece.data_type),
        converted_value=piece.converted_value or piece.original_value,
        converted_value_data_type=cast(PromptDataType, piece.data_type),
        conversation_id=conversation_id,
        sequence=sequence,
        prompt_metadata=metadata,
        labels=labels or {},
        original_prompt_id=original_prompt_id,
    )


def request_to_pyrit_message(
    *,
    request: AddMessageRequest,
    conversation_id: str,
    sequence: int,
    labels: Optional[Dict[str, str]] = None,
) -> PyritMessage:
    """
    Build a PyRIT Message from an AddMessageRequest DTO.

    Args:
        request: The inbound API request.
        conversation_id: The conversation/attack ID.
        sequence: The message sequence number.
        labels: Optional labels to stamp on each piece.

    Returns:
        PyritMessage ready to send to the target.
    """
    pieces = [
        request_piece_to_pyrit_message_piece(
            piece=p,
            role=request.role,
            conversation_id=conversation_id,
            sequence=sequence,
            labels=labels,
        )
        for p in request.pieces
    ]
    return PyritMessage(pieces)


# ============================================================================
# Private Helpers
# ============================================================================


def _get_preview_from_pieces(pieces: Sequence[PyritMessagePiece]) -> Optional[str]:
    """
    Get a preview of the last message from a list of pieces.

    Returns:
        Truncated last message text, or None if no pieces.
    """
    if not pieces:
        return None
    last_piece = max(pieces, key=lambda p: p.sequence)
    text = last_piece.converted_value or ""
    return text[:100] + "..." if len(text) > 100 else text


def _collect_labels_from_pieces(pieces: Sequence[PyritMessagePiece]) -> Dict[str, str]:
    """
    Collect labels from message pieces.

    Returns the labels from the first piece that has non-empty labels.
    All pieces in an attack share the same labels, so the first match
    is representative.

    Returns:
        Label dict, or empty dict if no pieces have labels.
    """
    for p in pieces:
        if p.labels:
            return dict(p.labels)
    return {}
