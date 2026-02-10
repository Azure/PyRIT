# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
Tests for backend mapper functions.

These tests verify the domain ↔ DTO translation layer in isolation,
without any database or service dependencies.
"""

from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest

from pyrit.backend.mappers.attack_mappers import (
    attack_result_to_summary,
    map_outcome,
    pyrit_messages_to_dto,
    pyrit_scores_to_dto,
    request_piece_to_pyrit_message_piece,
    request_to_pyrit_message,
)
from pyrit.backend.mappers.converter_mappers import converter_object_to_instance
from pyrit.backend.mappers.target_mappers import target_object_to_instance
from pyrit.models import AttackOutcome, AttackResult


# ============================================================================
# Helpers
# ============================================================================


def _make_attack_result(
    *,
    conversation_id: str = "attack-1",
    target_id: str = "target-1",
    target_type: str = "TextTarget",
    name: str = "Test Attack",
    outcome: AttackOutcome = AttackOutcome.UNDETERMINED,
    labels: dict = None,
) -> AttackResult:
    """Create an AttackResult for mapper tests."""
    now = datetime.now(timezone.utc)
    return AttackResult(
        conversation_id=conversation_id,
        objective="test",
        attack_identifier={
            "name": name,
            "target_id": target_id,
            "target_type": target_type,
        },
        outcome=outcome,
        metadata={
            "created_at": now.isoformat(),
            "updated_at": now.isoformat(),
            "labels": labels or {},
        },
    )


def _make_mock_piece(
    *,
    sequence: int = 0,
    converted_value: str = "hello",
    original_value: str = "hello",
):
    """Create a mock message piece for mapper tests."""
    p = MagicMock()
    p.id = "piece-1"
    p.sequence = sequence
    p.converted_value = converted_value
    p.original_value = original_value
    p.converted_value_data_type = "text"
    p.response_error = "none"
    p.role = "user"
    p.timestamp = datetime.now(timezone.utc)
    p.scores = []
    return p


def _make_mock_score():
    """Create a mock score for mapper tests."""
    s = MagicMock()
    s.id = "score-1"
    s.scorer_class_identifier = {"__type__": "TrueFalseScorer"}
    s.score_value = 1.0
    s.score_rationale = "Looks correct"
    s.timestamp = datetime.now(timezone.utc)
    return s


# ============================================================================
# Attack Mapper Tests
# ============================================================================


class TestMapOutcome:
    """Tests for map_outcome function."""

    def test_maps_success(self) -> None:
        assert map_outcome(AttackOutcome.SUCCESS) == "success"

    def test_maps_failure(self) -> None:
        assert map_outcome(AttackOutcome.FAILURE) == "failure"

    def test_maps_undetermined(self) -> None:
        assert map_outcome(AttackOutcome.UNDETERMINED) == "undetermined"


class TestAttackResultToSummary:
    """Tests for attack_result_to_summary function."""

    def test_basic_mapping(self) -> None:
        """Test that all fields are mapped correctly."""
        ar = _make_attack_result(name="My Attack", target_id="t-1", target_type="OpenAIChatTarget")
        pieces = [_make_mock_piece(sequence=0), _make_mock_piece(sequence=1)]

        summary = attack_result_to_summary(ar, pieces=pieces)

        assert summary.attack_id == ar.conversation_id
        assert summary.name == "My Attack"
        assert summary.target_id == "t-1"
        assert summary.target_type == "OpenAIChatTarget"
        assert summary.outcome == "undetermined"
        assert summary.message_count == 2

    def test_empty_pieces_gives_zero_messages(self) -> None:
        """Test mapping with no message pieces."""
        ar = _make_attack_result()

        summary = attack_result_to_summary(ar, pieces=[])

        assert summary.message_count == 0
        assert summary.last_message_preview is None

    def test_last_message_preview_truncated(self) -> None:
        """Test that long messages are truncated to 100 chars + ellipsis."""
        ar = _make_attack_result()
        long_text = "x" * 200
        pieces = [_make_mock_piece(converted_value=long_text)]

        summary = attack_result_to_summary(ar, pieces=pieces)

        assert summary.last_message_preview is not None
        assert len(summary.last_message_preview) == 103  # 100 + "..."
        assert summary.last_message_preview.endswith("...")

    def test_labels_are_mapped(self) -> None:
        """Test that labels are extracted from metadata."""
        ar = _make_attack_result(labels={"env": "prod", "team": "red"})

        summary = attack_result_to_summary(ar, pieces=[])

        assert summary.labels == {"env": "prod", "team": "red"}

    def test_outcome_success(self) -> None:
        """Test that success outcome is mapped."""
        ar = _make_attack_result(outcome=AttackOutcome.SUCCESS)

        summary = attack_result_to_summary(ar, pieces=[])

        assert summary.outcome == "success"


class TestPyritScoresToDto:
    """Tests for pyrit_scores_to_dto function."""

    def test_maps_scores(self) -> None:
        """Test that scores are correctly translated."""
        mock_score = _make_mock_score()

        result = pyrit_scores_to_dto([mock_score])

        assert len(result) == 1
        assert result[0].score_id == "score-1"
        assert result[0].scorer_type == "TrueFalseScorer"
        assert result[0].score_value == 1.0
        assert result[0].score_rationale == "Looks correct"

    def test_empty_scores(self) -> None:
        """Test mapping empty scores list."""
        result = pyrit_scores_to_dto([])
        assert result == []


class TestPyritMessagesToDto:
    """Tests for pyrit_messages_to_dto function."""

    def test_maps_single_message(self) -> None:
        """Test mapping a single message with one piece."""
        piece = _make_mock_piece(original_value="hi", converted_value="hi")
        msg = MagicMock()
        msg.message_pieces = [piece]

        result = pyrit_messages_to_dto([msg])

        assert len(result) == 1
        assert result[0].role == "user"
        assert len(result[0].pieces) == 1
        assert result[0].pieces[0].original_value == "hi"
        assert result[0].pieces[0].converted_value == "hi"

    def test_maps_empty_list(self) -> None:
        """Test mapping an empty messages list."""
        result = pyrit_messages_to_dto([])
        assert result == []


class TestRequestToPyritMessage:
    """Tests for request_to_pyrit_message function."""

    def test_converts_request_to_domain(self) -> None:
        """Test that DTO request is correctly converted to domain message."""
        request = MagicMock()
        request.role = "user"
        piece = MagicMock()
        piece.data_type = "text"
        piece.original_value = "hello"
        piece.converted_value = None
        request.pieces = [piece]

        result = request_to_pyrit_message(
            request=request,
            conversation_id="conv-1",
            sequence=0,
        )

        assert len(result.message_pieces) == 1
        assert result.message_pieces[0].original_value == "hello"
        assert result.message_pieces[0].conversation_id == "conv-1"
        assert result.message_pieces[0].sequence == 0


class TestRequestPieceToPyritMessagePiece:
    """Tests for request_piece_to_pyrit_message_piece function."""

    def test_uses_converted_value_when_present(self) -> None:
        """Test that converted_value is used when provided."""
        piece = MagicMock()
        piece.data_type = "text"
        piece.original_value = "original"
        piece.converted_value = "converted"

        result = request_piece_to_pyrit_message_piece(
            piece=piece,
            role="assistant",
            conversation_id="conv-1",
            sequence=5,
        )

        assert result.original_value == "original"
        assert result.converted_value == "converted"
        assert result.api_role == "assistant"
        assert result.sequence == 5

    def test_falls_back_to_original_when_no_converted(self) -> None:
        """Test that original_value is used when converted_value is None."""
        piece = MagicMock()
        piece.data_type = "text"
        piece.original_value = "fallback"
        piece.converted_value = None

        result = request_piece_to_pyrit_message_piece(
            piece=piece,
            role="user",
            conversation_id="conv-1",
            sequence=0,
        )

        assert result.converted_value == "fallback"


# ============================================================================
# Target Mapper Tests
# ============================================================================


class TestTargetObjectToInstance:
    """Tests for target_object_to_instance function."""

    def test_maps_target_with_identifier(self) -> None:
        """Test mapping a target object that has get_identifier."""
        target_obj = MagicMock()
        target_obj.get_identifier.return_value = {"__type__": "OpenAIChatTarget", "endpoint": "http://test"}

        result = target_object_to_instance("t-1", target_obj)

        assert result.target_id == "t-1"
        assert result.type == "OpenAIChatTarget"
        assert result.display_name is None

    def test_filters_sensitive_fields(self) -> None:
        """Test that sensitive fields are removed from params."""
        target_obj = MagicMock()
        target_obj.get_identifier.return_value = {
            "__type__": "TestTarget",
            "api_key": "secret-key",
            "endpoint": "http://test",
        }

        result = target_object_to_instance("t-1", target_obj)

        assert "api_key" not in result.params
        assert result.params.get("endpoint") == "http://test"

    def test_fallback_to_class_name(self) -> None:
        """Test fallback to __class__.__name__ when no __type__ in identifier."""
        target_obj = MagicMock()
        target_obj.__class__.__name__ = "FallbackTarget"
        target_obj.get_identifier.return_value = {"endpoint": "http://test"}

        result = target_object_to_instance("t-1", target_obj)

        assert result.type == "FallbackTarget"


# ============================================================================
# Converter Mapper Tests
# ============================================================================


class TestConverterObjectToInstance:
    """Tests for converter_object_to_instance function."""

    def test_maps_converter_with_identifier(self) -> None:
        """Test mapping a converter object."""
        converter_obj = MagicMock()
        identifier = MagicMock()
        identifier.to_dict.return_value = {"class_name": "Base64Converter", "param1": "value1"}
        converter_obj.get_identifier.return_value = identifier

        result = converter_object_to_instance("c-1", converter_obj)

        assert result.converter_id == "c-1"
        assert result.type == "Base64Converter"
        assert result.display_name is None
        assert result.params["class_name"] == "Base64Converter"

    def test_fallback_to_class_name(self) -> None:
        """Test fallback to __class__.__name__ when no class_name in identifier."""
        converter_obj = MagicMock()
        converter_obj.__class__.__name__ = "FallbackConverter"
        identifier = MagicMock()
        identifier.to_dict.return_value = {"param1": "value1"}
        converter_obj.get_identifier.return_value = identifier

        result = converter_object_to_instance("c-1", converter_obj)

        assert result.type == "FallbackConverter"
