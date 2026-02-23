# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
Tests for backend mapper functions.

These tests verify the domain ↔ DTO translation layer in isolation,
without any database or service dependencies.
"""

from datetime import datetime, timezone
from unittest.mock import MagicMock

from pyrit.backend.mappers.attack_mappers import (
    _collect_labels_from_pieces,
    _infer_mime_type,
    attack_result_to_summary,
    pyrit_messages_to_dto,
    pyrit_scores_to_dto,
    request_piece_to_pyrit_message_piece,
    request_to_pyrit_message,
)
from pyrit.backend.mappers.converter_mappers import converter_object_to_instance
from pyrit.backend.mappers.target_mappers import target_object_to_instance
from pyrit.identifiers import ComponentIdentifier
from pyrit.models import AttackOutcome, AttackResult

# ============================================================================
# Helpers
# ============================================================================


def _make_attack_result(
    *,
    conversation_id: str = "attack-1",
    has_target: bool = True,
    name: str = "Test Attack",
    outcome: AttackOutcome = AttackOutcome.UNDETERMINED,
) -> AttackResult:
    """Create an AttackResult for mapper tests."""
    now = datetime.now(timezone.utc)

    target_identifier = (
        ComponentIdentifier(
            class_name="TextTarget",
            class_module="pyrit.prompt_target",
        )
        if has_target
        else None
    )

    children = {}
    if target_identifier:
        children["objective_target"] = target_identifier

    return AttackResult(
        conversation_id=conversation_id,
        objective="test",
        attack_identifier=ComponentIdentifier(
            class_name=name,
            class_module="pyrit.backend",
            params={
                "source": "gui",
            },
            children=children,
        ),
        outcome=outcome,
        metadata={
            "created_at": now.isoformat(),
            "updated_at": now.isoformat(),
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
    p.original_value_data_type = "text"
    p.response_error = "none"
    p.role = "user"
    p.get_role_for_storage.return_value = "user"
    p.timestamp = datetime.now(timezone.utc)
    p.scores = []
    return p


def _make_mock_score():
    """Create a mock score for mapper tests."""
    s = MagicMock()
    s.id = "score-1"
    s.scorer_class_identifier = ComponentIdentifier(
        class_name="TrueFalseScorer",
        class_module="pyrit.score",
        params={"scorer_type": "true_false"},
    )
    s.score_value = 1.0
    s.score_rationale = "Looks correct"
    s.timestamp = datetime.now(timezone.utc)
    return s


# ============================================================================
# Attack Mapper Tests
# ============================================================================


class TestAttackResultToSummary:
    """Tests for attack_result_to_summary function."""

    def test_basic_mapping(self) -> None:
        """Test that all fields are mapped correctly."""
        ar = _make_attack_result(name="My Attack")
        pieces = [_make_mock_piece(sequence=0), _make_mock_piece(sequence=1)]

        summary = attack_result_to_summary(ar, pieces=pieces)

        assert summary.conversation_id == ar.conversation_id
        assert summary.outcome == "undetermined"
        assert summary.message_count == 2
        # Attack metadata should be extracted into explicit fields
        assert summary.attack_type == "My Attack"
        assert summary.target_type == "TextTarget"
        assert summary.target_unique_name is not None

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
        """Test that labels are derived from pieces."""
        ar = _make_attack_result()
        piece = _make_mock_piece()
        piece.labels = {"env": "prod", "team": "red"}

        summary = attack_result_to_summary(ar, pieces=[piece])

        assert summary.labels == {"env": "prod", "team": "red"}

    def test_outcome_success(self) -> None:
        """Test that success outcome is mapped."""
        ar = _make_attack_result(outcome=AttackOutcome.SUCCESS)

        summary = attack_result_to_summary(ar, pieces=[])

        assert summary.outcome == "success"

    def test_no_target_returns_none_fields(self) -> None:
        """Test that target fields are None when no target identifier exists."""
        ar = _make_attack_result(has_target=False)

        summary = attack_result_to_summary(ar, pieces=[])

        assert summary.target_unique_name is None
        assert summary.target_type is None

    def test_attack_specific_params_passed_through(self) -> None:
        """Test that attack_specific_params are extracted from identifier."""
        ar = _make_attack_result()

        summary = attack_result_to_summary(ar, pieces=[])

        assert summary.attack_specific_params == {"source": "gui"}

    def test_converters_extracted_from_identifier(self) -> None:
        """Test that converter class names are extracted into converters list."""
        now = datetime.now(timezone.utc)
        ar = AttackResult(
            conversation_id="attack-conv",
            objective="test",
            attack_identifier=ComponentIdentifier(
                class_name="TestAttack",
                class_module="pyrit.backend",
                children={
                    "request_converters": [
                        ComponentIdentifier(
                            class_name="Base64Converter",
                            class_module="pyrit.converters",
                            params={
                                "supported_input_types": ("text",),
                                "supported_output_types": ("text",),
                            },
                        ),
                        ComponentIdentifier(
                            class_name="ROT13Converter",
                            class_module="pyrit.converters",
                            params={
                                "supported_input_types": ("text",),
                                "supported_output_types": ("text",),
                            },
                        ),
                    ],
                },
            ),
            outcome=AttackOutcome.UNDETERMINED,
            metadata={"created_at": now.isoformat(), "updated_at": now.isoformat()},
        )

        summary = attack_result_to_summary(ar, pieces=[])

        assert summary.converters == ["Base64Converter", "ROT13Converter"]

    def test_no_converters_returns_empty_list(self) -> None:
        """Test that converters is empty list when no converters in identifier."""
        ar = _make_attack_result()

        summary = attack_result_to_summary(ar, pieces=[])

        assert summary.converters == []


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

    def test_maps_data_types_separately(self) -> None:
        """Test that original and converted data types are mapped independently."""
        piece = _make_mock_piece(original_value="describe this", converted_value="base64data")
        piece.original_value_data_type = "text"
        piece.converted_value_data_type = "image"
        msg = MagicMock()
        msg.message_pieces = [piece]

        result = pyrit_messages_to_dto([msg])

        assert result[0].pieces[0].original_value_data_type == "text"
        assert result[0].pieces[0].converted_value_data_type == "image"

    def test_maps_empty_list(self) -> None:
        """Test mapping an empty messages list."""
        result = pyrit_messages_to_dto([])
        assert result == []

    def test_populates_mime_type_for_image(self) -> None:
        """Test that MIME types are inferred for image pieces."""
        piece = _make_mock_piece(original_value="/path/to/photo.png", converted_value="/path/to/photo.jpg")
        piece.original_value_data_type = "image"
        piece.converted_value_data_type = "image"
        msg = MagicMock()
        msg.message_pieces = [piece]

        result = pyrit_messages_to_dto([msg])

        assert result[0].pieces[0].original_value_mime_type == "image/png"
        assert result[0].pieces[0].converted_value_mime_type == "image/jpeg"

    def test_mime_type_none_for_text(self) -> None:
        """Test that MIME type is None for text pieces."""
        piece = _make_mock_piece(original_value="hello", converted_value="hello")
        msg = MagicMock()
        msg.message_pieces = [piece]

        result = pyrit_messages_to_dto([msg])

        assert result[0].pieces[0].original_value_mime_type is None
        assert result[0].pieces[0].converted_value_mime_type is None

    def test_mime_type_for_audio(self) -> None:
        """Test that MIME types are inferred for audio pieces."""
        piece = _make_mock_piece(original_value="/tmp/speech.wav", converted_value="/tmp/speech.mp3")
        piece.original_value_data_type = "audio"
        piece.converted_value_data_type = "audio"
        msg = MagicMock()
        msg.message_pieces = [piece]

        result = pyrit_messages_to_dto([msg])

        # Python 3.10 returns "audio/wav", 3.11+ returns "audio/x-wav"
        assert result[0].pieces[0].original_value_mime_type in ("audio/wav", "audio/x-wav")
        assert result[0].pieces[0].converted_value_mime_type == "audio/mpeg"


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
        piece.original_prompt_id = None
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
        piece.original_prompt_id = None

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
        piece.original_prompt_id = None

        result = request_piece_to_pyrit_message_piece(
            piece=piece,
            role="user",
            conversation_id="conv-1",
            sequence=0,
        )

        assert result.converted_value == "fallback"

    def test_passes_mime_type_through_prompt_metadata(self) -> None:
        """Test that mime_type is stored in prompt_metadata."""
        piece = MagicMock()
        piece.data_type = "image_path"
        piece.original_value = "base64data"
        piece.converted_value = None
        piece.mime_type = "image/png"
        piece.original_prompt_id = None

        result = request_piece_to_pyrit_message_piece(
            piece=piece,
            role="user",
            conversation_id="conv-1",
            sequence=0,
        )

        assert result.prompt_metadata == {"mime_type": "image/png"}

    def test_no_metadata_when_mime_type_absent(self) -> None:
        """Test that prompt_metadata is empty when mime_type is None."""
        piece = MagicMock()
        piece.data_type = "text"
        piece.original_value = "hello"
        piece.converted_value = None
        piece.mime_type = None
        piece.original_prompt_id = None

        result = request_piece_to_pyrit_message_piece(
            piece=piece,
            role="user",
            conversation_id="conv-1",
            sequence=0,
        )

        assert result.prompt_metadata == {}

    def test_labels_are_stamped_on_piece(self) -> None:
        """Test that labels are passed through to the MessagePiece."""
        piece = MagicMock()
        piece.data_type = "text"
        piece.original_value = "hello"
        piece.converted_value = None
        piece.mime_type = None
        piece.original_prompt_id = None

        result = request_piece_to_pyrit_message_piece(
            piece=piece,
            role="user",
            conversation_id="conv-1",
            sequence=0,
            labels={"env": "prod"},
        )

        assert result.labels == {"env": "prod"}

    def test_labels_default_to_empty_dict(self) -> None:
        """Test that labels default to empty dict when not provided."""
        piece = MagicMock()
        piece.data_type = "text"
        piece.original_value = "hello"
        piece.converted_value = None
        piece.mime_type = None
        piece.original_prompt_id = None

        result = request_piece_to_pyrit_message_piece(
            piece=piece,
            role="user",
            conversation_id="conv-1",
            sequence=0,
        )

        assert result.labels == {}

    def test_original_prompt_id_forwarded_when_provided(self) -> None:
        """Test that original_prompt_id is passed through for lineage tracking."""
        piece = MagicMock()
        piece.data_type = "text"
        piece.original_value = "hello"
        piece.converted_value = None
        piece.mime_type = None
        piece.original_prompt_id = "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"

        result = request_piece_to_pyrit_message_piece(
            piece=piece,
            role="user",
            conversation_id="conv-1",
            sequence=0,
        )

        import uuid

        assert result.original_prompt_id == uuid.UUID("aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee")
        # New piece should have its own id, different from original_prompt_id
        assert result.id != result.original_prompt_id

    def test_original_prompt_id_defaults_to_self_when_absent(self) -> None:
        """Test that original_prompt_id defaults to the piece's own id when not provided."""
        piece = MagicMock()
        piece.data_type = "text"
        piece.original_value = "hello"
        piece.converted_value = None
        piece.mime_type = None
        piece.original_prompt_id = None

        result = request_piece_to_pyrit_message_piece(
            piece=piece,
            role="user",
            conversation_id="conv-1",
            sequence=0,
        )

        assert result.original_prompt_id == result.id


class TestInferMimeType:
    """Tests for _infer_mime_type helper function."""

    def test_returns_none_for_text(self) -> None:
        """Text data type should always return None."""
        assert _infer_mime_type(value="/path/to/file.png", data_type="text") is None

    def test_returns_none_for_empty_value(self) -> None:
        """Empty or None value should return None."""
        assert _infer_mime_type(value=None, data_type="image") is None
        assert _infer_mime_type(value="", data_type="image") is None

    def test_infers_png(self) -> None:
        """Test MIME type inference for PNG files."""
        assert _infer_mime_type(value="/tmp/photo.png", data_type="image") == "image/png"

    def test_infers_jpeg(self) -> None:
        """Test MIME type inference for JPEG files."""
        assert _infer_mime_type(value="/tmp/photo.jpg", data_type="image") == "image/jpeg"

    def test_infers_wav(self) -> None:
        """Test MIME type inference for WAV files."""
        result = _infer_mime_type(value="/tmp/audio.wav", data_type="audio")
        assert result is not None
        assert "wav" in result

    def test_infers_mp3(self) -> None:
        """Test MIME type inference for MP3 files."""
        assert _infer_mime_type(value="/tmp/audio.mp3", data_type="audio") == "audio/mpeg"

    def test_returns_none_for_unknown_extension(self) -> None:
        """Test that unrecognized extensions return None."""
        assert _infer_mime_type(value="/tmp/data.xyz123", data_type="image") is None

    def test_infers_mp4(self) -> None:
        """Test MIME type inference for MP4 video files."""
        assert _infer_mime_type(value="/tmp/video.mp4", data_type="video") == "video/mp4"


class TestCollectLabelsFromPieces:
    """Tests for _collect_labels_from_pieces helper."""

    def test_returns_labels_from_first_piece(self) -> None:
        """Returns labels from the first piece that has them."""
        p1 = MagicMock()
        p1.labels = {"env": "prod"}
        p2 = MagicMock()
        p2.labels = {"env": "staging"}

        assert _collect_labels_from_pieces([p1, p2]) == {"env": "prod"}

    def test_returns_empty_when_no_pieces(self) -> None:
        """Returns empty dict for empty list."""
        assert _collect_labels_from_pieces([]) == {}

    def test_returns_empty_when_pieces_have_no_labels(self) -> None:
        """Returns empty dict when pieces have None/empty labels."""
        p = MagicMock()
        p.labels = None
        assert _collect_labels_from_pieces([p]) == {}

    def test_skips_pieces_with_empty_labels(self) -> None:
        """Skips pieces with empty/falsy labels."""
        p1 = MagicMock()
        p1.labels = {}
        p2 = MagicMock()
        p2.labels = {"env": "prod"}

        assert _collect_labels_from_pieces([p1, p2]) == {"env": "prod"}


# ============================================================================
# Target Mapper Tests
# ============================================================================


class TestTargetObjectToInstance:
    """Tests for target_object_to_instance function."""

    def test_maps_target_with_identifier(self) -> None:
        """Test mapping a target object that has get_identifier."""
        target_obj = MagicMock()
        mock_identifier = ComponentIdentifier(
            class_name="OpenAIChatTarget",
            class_module="pyrit.prompt_target",
            params={
                "endpoint": "http://test",
                "model_name": "gpt-4",
                "temperature": 0.7,
            },
        )
        target_obj.get_identifier.return_value = mock_identifier

        result = target_object_to_instance("t-1", target_obj)

        assert result.target_unique_name == "t-1"
        assert result.target_type == "OpenAIChatTarget"
        assert result.endpoint == "http://test"
        assert result.model_name == "gpt-4"
        assert result.temperature == 0.7

    def test_no_endpoint_returns_none(self) -> None:
        """Test that missing endpoint returns None."""
        target_obj = MagicMock()
        mock_identifier = ComponentIdentifier(
            class_name="TextTarget",
            class_module="pyrit.prompt_target",
        )
        target_obj.get_identifier.return_value = mock_identifier

        result = target_object_to_instance("t-1", target_obj)

        assert result.target_type == "TextTarget"
        assert result.endpoint is None
        assert result.model_name is None

    def test_no_get_identifier_uses_class_name(self) -> None:
        """Test that target uses class name from identifier."""
        target_obj = MagicMock()
        mock_identifier = ComponentIdentifier(class_name="FakeTarget", class_module="pyrit.prompt_target")
        target_obj.get_identifier.return_value = mock_identifier

        result = target_object_to_instance("t-1", target_obj)

        assert result.target_type == "FakeTarget"
        assert result.endpoint is None
        assert result.model_name is None


# ============================================================================
# Converter Mapper Tests
# ============================================================================


class TestConverterObjectToInstance:
    """Tests for converter_object_to_instance function."""

    def test_maps_converter_with_identifier(self) -> None:
        """Test mapping a converter object."""
        converter_obj = MagicMock()
        identifier = ComponentIdentifier(
            class_name="Base64Converter",
            class_module="pyrit.converters",
            params={
                "supported_input_types": ("text",),
                "supported_output_types": ("text",),
                "param1": "value1",
            },
        )
        converter_obj.get_identifier.return_value = identifier

        result = converter_object_to_instance("c-1", converter_obj)

        assert result.converter_id == "c-1"
        assert result.converter_type == "Base64Converter"
        assert result.display_name is None
        assert result.supported_input_types == ["text"]
        assert result.supported_output_types == ["text"]
        assert result.converter_specific_params == {"param1": "value1"}
        assert result.sub_converter_ids is None

    def test_sub_converter_ids_passed_through(self) -> None:
        """Test that sub_converter_ids are passed through when provided."""
        converter_obj = MagicMock()
        identifier = ComponentIdentifier(
            class_name="PipelineConverter",
            class_module="pyrit.converters",
            params={
                "supported_input_types": ("text",),
                "supported_output_types": ("text",),
            },
        )
        converter_obj.get_identifier.return_value = identifier

        result = converter_object_to_instance("c-1", converter_obj, sub_converter_ids=["sub-1", "sub-2"])

        assert result.sub_converter_ids == ["sub-1", "sub-2"]

    def test_none_input_output_types_returns_empty_lists(self) -> None:
        """Test that None supported types produce empty lists."""
        converter_obj = MagicMock()
        identifier = ComponentIdentifier(
            class_name="CustomConverter",
            class_module="pyrit.converters",
        )
        converter_obj.get_identifier.return_value = identifier

        result = converter_object_to_instance("c-1", converter_obj)

        assert result.supported_input_types == []
        assert result.supported_output_types == []
        assert result.converter_specific_params is None
        assert result.sub_converter_ids is None
