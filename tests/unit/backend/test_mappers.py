# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
Tests for backend mapper functions.

These tests verify the domain ↔ DTO translation layer in isolation,
without any database or service dependencies.
"""

import dataclasses
import os
import tempfile
import uuid
import pytest
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

from pyrit.backend.mappers.attack_mappers import (
    _build_filename,
    _fetch_blob_as_data_uri_async,
    _infer_mime_type,
    _is_azure_blob_url,
    _sign_blob_url_async,
    attack_result_to_summary,
    pyrit_messages_to_dto_async,
    pyrit_scores_to_dto,
    request_piece_to_pyrit_message_piece,
    request_to_pyrit_message,
)
from pyrit.backend.mappers.converter_mappers import converter_object_to_instance
from pyrit.backend.mappers.target_mappers import target_object_to_instance
from pyrit.identifiers import ComponentIdentifier
from pyrit.models import AttackOutcome, AttackResult
from pyrit.models.conversation_stats import ConversationStats

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
        stats = ConversationStats(message_count=2)

        summary = attack_result_to_summary(ar, stats=stats)

        assert summary.conversation_id == ar.conversation_id
        assert summary.outcome == "undetermined"
        assert summary.message_count == 2
        # Attack metadata should be extracted into explicit fields
        assert summary.attack_type == "My Attack"
        assert summary.target is not None
        assert summary.target.target_type == "TextTarget"

    def test_empty_pieces_gives_zero_messages(self) -> None:
        """Test mapping with no message pieces."""
        ar = _make_attack_result()
        stats = ConversationStats(message_count=0)

        summary = attack_result_to_summary(ar, stats=stats)

        assert summary.message_count == 0
        assert summary.last_message_preview is None

    def test_last_message_preview_truncated(self) -> None:
        """Test that long messages are truncated in stats."""
        ar = _make_attack_result()
        long_text = "x" * 200
        stats = ConversationStats(message_count=1, last_message_preview=long_text[:100] + "...")

        summary = attack_result_to_summary(ar, stats=stats)

        assert summary.last_message_preview is not None
        assert len(summary.last_message_preview) == 103  # 100 + "..."
        assert summary.last_message_preview.endswith("...")

    def test_labels_are_mapped(self) -> None:
        """Test that labels are derived from stats."""
        ar = _make_attack_result()
        stats = ConversationStats(message_count=1, labels={"env": "prod", "team": "red"})

        summary = attack_result_to_summary(ar, stats=stats)

        assert summary.labels == {"env": "prod", "team": "red"}

    def test_labels_passed_through_without_normalization(self) -> None:
        """Test that labels are passed through as-is (DB stores canonical keys after migration)."""
        ar = _make_attack_result()
        stats = ConversationStats(
            message_count=1,
            labels={"operator": "alice", "operation": "op_red", "env": "prod"},
        )

        summary = attack_result_to_summary(ar, stats=stats)

        assert summary.labels == {"operator": "alice", "operation": "op_red", "env": "prod"}

    def test_outcome_success(self) -> None:
        """Test that success outcome is mapped."""
        ar = _make_attack_result(outcome=AttackOutcome.SUCCESS)
        stats = ConversationStats(message_count=0)

        summary = attack_result_to_summary(ar, stats=stats)

        assert summary.outcome == "success"

    def test_no_target_returns_none_fields(self) -> None:
        """Test that target fields are None when no target identifier exists."""
        ar = _make_attack_result(has_target=False)
        stats = ConversationStats(message_count=0)

        summary = attack_result_to_summary(ar, stats=stats)

        assert summary.target is None

    def test_attack_specific_params_passed_through(self) -> None:
        """Test that attack_specific_params are extracted from identifier."""
        ar = _make_attack_result()
        stats = ConversationStats(message_count=0)

        summary = attack_result_to_summary(ar, stats=stats)

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

        summary = attack_result_to_summary(ar, stats=ConversationStats(message_count=0))

        assert summary.converters == ["Base64Converter", "ROT13Converter"]

    def test_no_converters_returns_empty_list(self) -> None:
        """Test that converters is empty list when no converters in identifier."""
        ar = _make_attack_result()
        stats = ConversationStats(message_count=0)

        summary = attack_result_to_summary(ar, stats=stats)

        assert summary.converters == []

    def test_related_conversation_ids_from_related_conversations(self) -> None:
        """Test that related_conversation_ids includes all related conversation IDs."""
        from pyrit.models.conversation_reference import ConversationReference, ConversationType

        ar = _make_attack_result()
        ar.related_conversations = {
            ConversationReference(
                conversation_id="branch-1",
                conversation_type=ConversationType.ADVERSARIAL,
            ),
            ConversationReference(
                conversation_id="pruned-1",
                conversation_type=ConversationType.PRUNED,
            ),
        }

        summary = attack_result_to_summary(ar, stats=ConversationStats(message_count=0))

        assert sorted(summary.related_conversation_ids) == ["branch-1", "pruned-1"]

    def test_related_conversation_ids_empty_when_no_related(self) -> None:
        """Test that related_conversation_ids is empty when no related conversations exist."""
        ar = _make_attack_result()
        stats = ConversationStats(message_count=0)

        summary = attack_result_to_summary(ar, stats=stats)

        assert summary.related_conversation_ids == []

    def test_message_count_from_stats(self) -> None:
        """Test that message_count comes from stats."""
        ar = _make_attack_result()
        stats = ConversationStats(message_count=5)

        summary = attack_result_to_summary(ar, stats=stats)

        assert summary.message_count == 5

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

    def test_invalid_score_values_are_skipped(self) -> None:
        """Test that non-numeric score values are ignored instead of raising."""
        valid_score = _make_mock_score()
        invalid_score = _make_mock_score()
        invalid_score.id = "score-invalid"
        invalid_score.score_value = "false"

        result = pyrit_scores_to_dto([valid_score, invalid_score])

        assert len(result) == 1
        assert result[0].score_id == "score-1"


class TestPyritMessagesToDto:
    """Tests for pyrit_messages_to_dto_async function."""

    @pytest.mark.asyncio
    async def test_maps_single_message(self) -> None:
        """Test mapping a single message with one piece."""
        piece = _make_mock_piece(original_value="hi", converted_value="hi")
        msg = MagicMock()
        msg.message_pieces = [piece]

        result = await pyrit_messages_to_dto_async([msg])

        assert len(result) == 1
        assert result[0].role == "user"
        assert len(result[0].pieces) == 1
        assert result[0].pieces[0].original_value == "hi"
        assert result[0].pieces[0].converted_value == "hi"

    @pytest.mark.asyncio
    async def test_maps_data_types_separately(self) -> None:
        """Test that original and converted data types are mapped independently."""
        piece = _make_mock_piece(original_value="describe this", converted_value="base64data")
        piece.original_value_data_type = "text"
        piece.converted_value_data_type = "image"
        msg = MagicMock()
        msg.message_pieces = [piece]

        result = await pyrit_messages_to_dto_async([msg])

        assert result[0].pieces[0].original_value_data_type == "text"
        assert result[0].pieces[0].converted_value_data_type == "image"

    @pytest.mark.asyncio
    async def test_maps_empty_list(self) -> None:
        """Test mapping an empty messages list."""
        result = await pyrit_messages_to_dto_async([])
        assert result == []

    @pytest.mark.asyncio
    async def test_populates_mime_type_for_image(self) -> None:
        """Test that MIME types are inferred for image pieces."""
        piece = _make_mock_piece(original_value="/path/to/photo.png", converted_value="/path/to/photo.jpg")
        piece.original_value_data_type = "image"
        piece.converted_value_data_type = "image"
        msg = MagicMock()
        msg.message_pieces = [piece]

        result = await pyrit_messages_to_dto_async([msg])

        assert result[0].pieces[0].original_value_mime_type == "image/png"
        assert result[0].pieces[0].converted_value_mime_type == "image/jpeg"

    @pytest.mark.asyncio
    async def test_mime_type_none_for_text(self) -> None:
        """Test that MIME type is None for text pieces."""
        piece = _make_mock_piece(original_value="hello", converted_value="hello")
        msg = MagicMock()
        msg.message_pieces = [piece]

        result = await pyrit_messages_to_dto_async([msg])

        assert result[0].pieces[0].original_value_mime_type is None
        assert result[0].pieces[0].converted_value_mime_type is None

    @pytest.mark.asyncio
    async def test_mime_type_for_audio(self) -> None:
        """Test that MIME types are inferred for audio pieces."""
        piece = _make_mock_piece(original_value="/tmp/speech.wav", converted_value="/tmp/speech.mp3")
        piece.original_value_data_type = "audio"
        piece.converted_value_data_type = "audio"
        msg = MagicMock()
        msg.message_pieces = [piece]

        result = await pyrit_messages_to_dto_async([msg])

        # Python 3.10 returns "audio/wav", 3.11+ returns "audio/x-wav"
        assert result[0].pieces[0].original_value_mime_type in ("audio/wav", "audio/x-wav")
        assert result[0].pieces[0].converted_value_mime_type == "audio/mpeg"

    @pytest.mark.asyncio
    async def test_encodes_existing_media_file_to_data_uri(self) -> None:
        """Test that local media files are base64-encoded into data URIs."""
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            tmp.write(b"PNGDATA")
            tmp_path = tmp.name

        try:
            piece = _make_mock_piece(original_value=tmp_path, converted_value=tmp_path)
            piece.original_value_data_type = "image_path"
            piece.converted_value_data_type = "image_path"
            msg = MagicMock()
            msg.message_pieces = [piece]

            result = await pyrit_messages_to_dto_async([msg])

            assert result[0].pieces[0].original_value is not None
            assert result[0].pieces[0].original_value.startswith("data:image/png;base64,")
            assert result[0].pieces[0].converted_value.startswith("data:image/png;base64,")
        finally:
            os.unlink(tmp_path)

    @pytest.mark.asyncio
    async def test_data_uri_passthrough(self) -> None:
        """Test that pre-encoded data URIs are not re-encoded."""
        piece = _make_mock_piece(
            original_value="data:image/png;base64,AAAA",
            converted_value="data:image/jpeg;base64,BBBB",
        )
        piece.original_value_data_type = "image_path"
        piece.converted_value_data_type = "image_path"
        msg = MagicMock()
        msg.message_pieces = [piece]

        result = await pyrit_messages_to_dto_async([msg])

        assert result[0].pieces[0].original_value == "data:image/png;base64,AAAA"
        assert result[0].pieces[0].converted_value == "data:image/jpeg;base64,BBBB"

    @pytest.mark.asyncio
    async def test_non_blob_http_url_passthrough(self) -> None:
        """Test that non-Azure-Blob HTTP URLs are passed through as-is."""
        piece = _make_mock_piece(
            original_value="http://example.com/image.png",
            converted_value="http://example.com/image.png",
        )
        piece.original_value_data_type = "image_path"
        piece.converted_value_data_type = "image_path"
        msg = MagicMock()
        msg.message_pieces = [piece]

        result = await pyrit_messages_to_dto_async([msg])

        assert result[0].pieces[0].original_value == "http://example.com/image.png"
        assert result[0].pieces[0].converted_value == "http://example.com/image.png"

    @pytest.mark.asyncio
    async def test_azure_blob_url_is_fetched_as_data_uri(self) -> None:
        """Test that Azure Blob Storage URLs are fetched and returned as data URIs."""
        blob_url = "https://myaccount.blob.core.windows.net/dbdata/prompt-memory-entries/images/test.png"
        piece = _make_mock_piece(
            original_value=blob_url,
            converted_value=blob_url,
        )
        piece.original_value_data_type = "image_path"
        piece.converted_value_data_type = "image_path"
        msg = MagicMock()
        msg.message_pieces = [piece]

        with patch(
            "pyrit.backend.mappers.attack_mappers._fetch_blob_as_data_uri_async",
            new_callable=AsyncMock,
            return_value="data:image/png;base64,ABCD",
        ):
            result = await pyrit_messages_to_dto_async([msg])

        assert result[0].pieces[0].original_value == "data:image/png;base64,ABCD"
        assert result[0].pieces[0].converted_value == "data:image/png;base64,ABCD"

    @pytest.mark.asyncio
    async def test_azure_blob_url_fetch_failure_returns_raw_url(self) -> None:
        """Test that blob fetch failure falls back to the raw blob URL."""
        blob_url = "https://myaccount.blob.core.windows.net/dbdata/images/test.png"
        piece = _make_mock_piece(
            original_value=blob_url,
            converted_value=blob_url,
        )
        piece.original_value_data_type = "image_path"
        piece.converted_value_data_type = "image_path"
        msg = MagicMock()
        msg.message_pieces = [piece]

        with patch(
            "pyrit.backend.mappers.attack_mappers._fetch_blob_as_data_uri_async",
            new_callable=AsyncMock,
            return_value=blob_url,  # falls back to raw URL
        ):
            result = await pyrit_messages_to_dto_async([msg])

        assert result[0].pieces[0].original_value == blob_url
        assert result[0].pieces[0].converted_value == blob_url

    @pytest.mark.asyncio
    async def test_media_read_failure_returns_raw_path(self) -> None:
        """Test that unreadable local media files fall back to raw path values."""
        piece = _make_mock_piece(original_value="/tmp/file.png", converted_value="/tmp/file.png")
        piece.original_value_data_type = "image_path"
        piece.converted_value_data_type = "image_path"
        msg = MagicMock()
        msg.message_pieces = [piece]

        with (
            patch("pyrit.backend.mappers.attack_mappers.os.path.isfile", return_value=True),
            patch("pyrit.backend.mappers.attack_mappers.open", side_effect=OSError("cannot read")),
        ):
            result = await pyrit_messages_to_dto_async([msg])

        assert result[0].pieces[0].original_value == "/tmp/file.png"
        assert result[0].pieces[0].converted_value == "/tmp/file.png"


class TestIsAzureBlobUrl:
    """Tests for _is_azure_blob_url helper."""

    def test_azure_blob_url_detected(self) -> None:
        assert _is_azure_blob_url("https://account.blob.core.windows.net/container/blob.png") is True

    def test_http_non_blob_url_not_detected(self) -> None:
        assert _is_azure_blob_url("http://example.com/image.png") is False

    def test_https_non_blob_url_not_detected(self) -> None:
        assert _is_azure_blob_url("https://example.com/image.png") is False

    def test_data_uri_not_detected(self) -> None:
        assert _is_azure_blob_url("data:image/png;base64,AAAA") is False

    def test_local_path_not_detected(self) -> None:
        assert _is_azure_blob_url("/tmp/test.png") is False


class TestSignBlobUrlAsync:
    """Tests for _sign_blob_url_async helper."""

    @pytest.mark.asyncio
    async def test_non_blob_url_unchanged(self) -> None:
        """Non-Azure URLs pass through without signing."""
        result = await _sign_blob_url_async(blob_url="http://example.com/img.png")
        assert result == "http://example.com/img.png"

    @pytest.mark.asyncio
    async def test_already_signed_url_unchanged(self) -> None:
        """URLs that already have query params (SAS) are not re-signed."""
        url = "https://acct.blob.core.windows.net/c/b.png?sv=2024&sig=abc"
        result = await _sign_blob_url_async(blob_url=url)
        assert result == url

    @pytest.mark.asyncio
    async def test_appends_sas_token(self) -> None:
        """SAS token is appended to unsigned blob URLs."""
        url = "https://acct.blob.core.windows.net/container/path/blob.png"
        with patch(
            "pyrit.backend.mappers.attack_mappers._get_sas_for_container_async",
            new_callable=AsyncMock,
            return_value="sv=2024&sig=test",
        ) as mock_sas:
            result = await _sign_blob_url_async(blob_url=url)

        assert result == f"{url}?sv=2024&sig=test"
        mock_sas.assert_called_once_with(container_url="https://acct.blob.core.windows.net/container")

    @pytest.mark.asyncio
    async def test_sas_failure_returns_original(self) -> None:
        """SAS generation failure falls back to the unsigned URL."""
        url = "https://acct.blob.core.windows.net/c/b.png"
        with patch(
            "pyrit.backend.mappers.attack_mappers._get_sas_for_container_async",
            new_callable=AsyncMock,
            side_effect=RuntimeError("auth error"),
        ):
            result = await _sign_blob_url_async(blob_url=url)

        assert result == url


class TestFetchBlobAsDataUriAsync:
    """Tests for _fetch_blob_as_data_uri_async helper."""

    @pytest.mark.asyncio
    async def test_fetches_blob_and_returns_data_uri(self) -> None:
        """Blob content is fetched, base64-encoded, and returned as a data URI."""
        import httpx

        blob_url = "https://acct.blob.core.windows.net/container/image.png"
        fake_resp = httpx.Response(
            status_code=200,
            content=b"\x89PNG",
            headers={"content-type": "image/png"},
            request=httpx.Request("GET", blob_url),
        )

        with (
            patch(
                "pyrit.backend.mappers.attack_mappers._sign_blob_url_async",
                new_callable=AsyncMock,
                return_value=blob_url + "?sig=abc",
            ),
            patch("pyrit.backend.mappers.attack_mappers.httpx.AsyncClient") as mock_client_cls,
        ):
            mock_client = AsyncMock()
            mock_client.get = AsyncMock(return_value=fake_resp)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            result = await _fetch_blob_as_data_uri_async(blob_url=blob_url)

        import base64

        expected_b64 = base64.b64encode(b"\x89PNG").decode("ascii")
        assert result == f"data:image/png;base64,{expected_b64}"

    @pytest.mark.asyncio
    async def test_fetch_failure_returns_raw_url(self) -> None:
        """Fetch failure falls back to the unsigned blob URL."""
        blob_url = "https://acct.blob.core.windows.net/container/file.wav"

        with (
            patch(
                "pyrit.backend.mappers.attack_mappers._sign_blob_url_async",
                new_callable=AsyncMock,
                return_value=blob_url + "?sig=abc",
            ),
            patch("pyrit.backend.mappers.attack_mappers.httpx.AsyncClient") as mock_client_cls,
        ):
            mock_client = AsyncMock()
            mock_client.get = AsyncMock(side_effect=Exception("network error"))
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            result = await _fetch_blob_as_data_uri_async(blob_url=blob_url)

        assert result == blob_url


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


class TestBuildFilename:
    """Tests for _build_filename helper function."""

    def test_image_path_with_hash(self) -> None:
        result = _build_filename(data_type="image_path", sha256="abcdef1234567890", value="/tmp/photo.png")
        assert result == "image_abcdef12.png"

    def test_audio_path_with_hash(self) -> None:
        result = _build_filename(data_type="audio_path", sha256="1234abcd5678efgh", value="/tmp/speech.wav")
        assert result == "audio_1234abcd.wav"

    def test_video_path_with_hash(self) -> None:
        result = _build_filename(data_type="video_path", sha256="deadbeef00000000", value="/tmp/clip.mp4")
        assert result == "video_deadbeef.mp4"

    def test_binary_path_with_hash(self) -> None:
        result = _build_filename(data_type="binary_path", sha256="cafe0123babe4567", value="/tmp/doc.pdf")
        assert result == "file_cafe0123.pdf"

    def test_returns_none_for_text(self) -> None:
        assert _build_filename(data_type="text", sha256="abc123", value="hello") is None

    def test_returns_none_for_reasoning(self) -> None:
        assert _build_filename(data_type="reasoning", sha256="abc123", value="thinking") is None

    def test_fallback_ext_when_no_value(self) -> None:
        result = _build_filename(data_type="image_path", sha256="abcdef1234567890", value=None)
        assert result == "image_abcdef12.png"

    def test_fallback_ext_for_data_uri(self) -> None:
        result = _build_filename(data_type="audio_path", sha256="abcdef1234567890", value="data:audio/wav;base64,AAA=")
        assert result == "audio_abcdef12.wav"

    def test_random_hash_when_no_sha256(self) -> None:
        result = _build_filename(data_type="image_path", sha256=None, value="/tmp/photo.png")
        assert result is not None
        assert result.startswith("image_")
        assert result.endswith(".png")
        assert len(result) == len("image_12345678.png")

    def test_blob_url_extension(self) -> None:
        url = "https://account.blob.core.windows.net/container/images/photo.jpg"
        result = _build_filename(data_type="image_path", sha256="abcdef1234567890", value=url)
        assert result == "image_abcdef12.jpg"


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

        assert result.target_registry_name == "t-1"
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

    def test_supports_multiturn_chat_true_for_prompt_chat_target(self) -> None:
        """Test that PromptChatTarget subclasses have supports_multiturn_chat=True."""
        from pyrit.prompt_target import PromptChatTarget

        target_obj = MagicMock(spec=PromptChatTarget)
        mock_identifier = ComponentIdentifier(
            class_name="OpenAIChatTarget",
            class_module="pyrit.prompt_target",
            params={
                "endpoint": "https://api.openai.com",
                "model_name": "gpt-4",
            },
        )
        target_obj.get_identifier.return_value = mock_identifier

        result = target_object_to_instance("t-1", target_obj)

        assert result.supports_multiturn_chat is True

    def test_supports_multiturn_chat_false_for_plain_prompt_target(self) -> None:
        """Test that plain PromptTarget (non-chat) has supports_multiturn_chat=False."""
        from pyrit.prompt_target import PromptTarget

        target_obj = MagicMock(spec=PromptTarget)
        mock_identifier = ComponentIdentifier(
            class_name="TextTarget",
            class_module="pyrit.prompt_target",
        )
        target_obj.get_identifier.return_value = mock_identifier

        result = target_object_to_instance("t-1", target_obj)

        assert result.supports_multiturn_chat is False


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



# ============================================================================
# Drift Detection Tests – verify mapper-accessed fields exist on domain models
# ============================================================================


class TestDomainModelFieldsExist:
    """Lightweight safety-net: ensure fields the mappers access still exist on the domain dataclasses.

    If a domain model field is renamed or removed, these tests fail immediately –
    before a mapper silently starts returning incorrect data.
    """

    # -- ComponentIdentifier fields used in attack_mappers.py -----------------

    @pytest.mark.parametrize(
        "field_name",
        [
            "class_name",
            "params",
            "children",
        ],
    )
    def test_component_identifier_has_field(self, field_name: str) -> None:
        field_names = {f.name for f in dataclasses.fields(ComponentIdentifier)}
        assert field_name in field_names, (
            f"ComponentIdentifier is missing '{field_name}' – mappers depend on this field"
        )
