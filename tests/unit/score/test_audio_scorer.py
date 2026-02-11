# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import tempfile
import uuid
from typing import Optional
from unittest.mock import AsyncMock, patch

import pytest

from pyrit.identifiers import ScorerIdentifier
from pyrit.models import MessagePiece, Score
from pyrit.score.float_scale.audio_float_scale_scorer import AudioFloatScaleScorer
from pyrit.score.float_scale.float_scale_scorer import FloatScaleScorer
from pyrit.score.scorer_prompt_validator import ScorerPromptValidator
from pyrit.score.true_false.audio_true_false_scorer import AudioTrueFalseScorer
from pyrit.score.true_false.true_false_scorer import TrueFalseScorer
from tests.unit.mocks import get_mock_scorer_identifier


class MockTextTrueFalseScorer(TrueFalseScorer):
    """Mock TrueFalseScorer for testing audio transcription scoring"""

    def __init__(self, return_value: bool = True):
        self.return_value = return_value
        validator = ScorerPromptValidator(supported_data_types=["text"])
        super().__init__(validator=validator)

    def _build_identifier(self) -> ScorerIdentifier:
        return self._create_identifier()

    async def _score_piece_async(self, message_piece: MessagePiece, *, objective: Optional[str] = None) -> list[Score]:
        return [
            Score(
                score_type="true_false",
                score_value=str(self.return_value).lower(),
                score_rationale=f"Test rationale for transcript: {message_piece.converted_value}",
                score_category=["test_category"],
                score_metadata={},
                score_value_description="test_description",
                message_piece_id=message_piece.id or uuid.uuid4(),
                objective=objective,
                scorer_class_identifier=get_mock_scorer_identifier(),
            )
        ]


class MockTextFloatScaleScorer(FloatScaleScorer):
    """Mock FloatScaleScorer for testing audio transcription scoring"""

    def __init__(self, return_value: float = 0.8):
        self.return_value = return_value
        validator = ScorerPromptValidator(supported_data_types=["text"])
        super().__init__(validator=validator)

    def _build_identifier(self) -> ScorerIdentifier:
        return self._create_identifier()

    async def _score_piece_async(self, message_piece: MessagePiece, *, objective: Optional[str] = None) -> list[Score]:
        return [
            Score(
                score_type="float_scale",
                score_value=str(self.return_value),
                score_rationale=f"Test rationale for transcript: {message_piece.converted_value}",
                score_category=["test_category"],
                score_metadata={},
                score_value_description="test_description",
                message_piece_id=message_piece.id or uuid.uuid4(),
                objective=objective,
                scorer_class_identifier=get_mock_scorer_identifier(),
            )
        ]


@pytest.fixture
def audio_message_piece(patch_central_database):
    """Create a mock audio message piece for testing"""
    # Create a temporary audio file
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
        temp_file.write(b"fake audio content")
        audio_path = temp_file.name

    message_piece = MessagePiece(
        role="user",
        original_value=audio_path,
        converted_value=audio_path,
        original_value_data_type="audio_path",
        converted_value_data_type="audio_path",
    )
    message_piece.id = uuid.uuid4()

    yield message_piece

    # Cleanup
    if os.path.exists(audio_path):
        os.remove(audio_path)


@pytest.mark.usefixtures("patch_central_database")
class TestAudioTrueFalseScorer:
    """Tests for AudioTrueFalseScorer"""

    def test_init_with_text_scorer(self):
        """Test initialization with a text-capable scorer"""
        text_scorer = MockTextTrueFalseScorer()
        audio_scorer = AudioTrueFalseScorer(text_capable_scorer=text_scorer)

        assert audio_scorer._audio_helper.text_scorer is text_scorer

    def test_build_identifier(self):
        """Test that _build_identifier returns correct identifier"""
        text_scorer = MockTextTrueFalseScorer()
        audio_scorer = AudioTrueFalseScorer(text_capable_scorer=text_scorer)

        identifier = audio_scorer._build_identifier()

        assert isinstance(identifier, ScorerIdentifier)

    @pytest.mark.asyncio
    async def test_score_piece_with_transcript(self, audio_message_piece):
        """Test scoring audio with a valid transcript"""
        text_scorer = MockTextTrueFalseScorer(return_value=True)
        audio_scorer = AudioTrueFalseScorer(text_capable_scorer=text_scorer)

        # Mock the transcription to return a test transcript
        with patch.object(
            audio_scorer._audio_helper, "_transcribe_audio_async", new_callable=AsyncMock
        ) as mock_transcribe:
            mock_transcribe.return_value = "Hello, this is a test transcript."

            scores = await audio_scorer._score_piece_async(audio_message_piece)

            assert len(scores) == 1
            assert scores[0].score_type == "true_false"
            assert scores[0].score_value == "true"
            assert "Audio transcript scored:" in scores[0].score_rationale

    @pytest.mark.asyncio
    async def test_score_piece_empty_transcript(self, audio_message_piece):
        """Test scoring audio with empty transcript returns empty list"""
        text_scorer = MockTextTrueFalseScorer(return_value=True)
        audio_scorer = AudioTrueFalseScorer(text_capable_scorer=text_scorer)

        # Mock the transcription to return empty string
        with patch.object(
            audio_scorer._audio_helper, "_transcribe_audio_async", new_callable=AsyncMock
        ) as mock_transcribe:
            mock_transcribe.return_value = ""

            scores = await audio_scorer._score_piece_async(audio_message_piece)

            # Empty transcript returns empty list
            assert len(scores) == 0

    @pytest.mark.asyncio
    async def test_score_piece_false_result(self, audio_message_piece):
        """Test scoring audio that returns false"""
        text_scorer = MockTextTrueFalseScorer(return_value=False)
        audio_scorer = AudioTrueFalseScorer(text_capable_scorer=text_scorer)

        # Mock the transcription
        with patch.object(
            audio_scorer._audio_helper, "_transcribe_audio_async", new_callable=AsyncMock
        ) as mock_transcribe:
            mock_transcribe.return_value = "Some transcript text"

            scores = await audio_scorer._score_piece_async(audio_message_piece)

            assert len(scores) == 1
            assert scores[0].score_type == "true_false"
            assert scores[0].score_value == "false"


@pytest.mark.usefixtures("patch_central_database")
class TestAudioFloatScaleScorer:
    """Tests for AudioFloatScaleScorer"""

    def test_init_with_text_scorer(self):
        """Test initialization with a text-capable scorer"""
        text_scorer = MockTextFloatScaleScorer()
        audio_scorer = AudioFloatScaleScorer(text_capable_scorer=text_scorer)

        assert audio_scorer._audio_helper.text_scorer is text_scorer

    def test_build_identifier(self):
        """Test that _build_identifier returns correct identifier"""
        text_scorer = MockTextFloatScaleScorer()
        audio_scorer = AudioFloatScaleScorer(text_capable_scorer=text_scorer)

        identifier = audio_scorer._build_identifier()

        assert isinstance(identifier, ScorerIdentifier)

    @pytest.mark.asyncio
    async def test_score_piece_with_transcript(self, audio_message_piece):
        """Test scoring audio with a valid transcript"""
        text_scorer = MockTextFloatScaleScorer(return_value=0.75)
        audio_scorer = AudioFloatScaleScorer(text_capable_scorer=text_scorer)

        # Mock the transcription to return a test transcript
        with patch.object(
            audio_scorer._audio_helper, "_transcribe_audio_async", new_callable=AsyncMock
        ) as mock_transcribe:
            mock_transcribe.return_value = "Hello, this is a test transcript."

            scores = await audio_scorer._score_piece_async(audio_message_piece)

            assert len(scores) == 1
            assert scores[0].score_type == "float_scale"
            assert float(scores[0].score_value) == 0.75
            assert "Audio transcript scored:" in scores[0].score_rationale

    @pytest.mark.asyncio
    async def test_score_piece_empty_transcript(self, audio_message_piece):
        """Test scoring audio with empty transcript returns empty list"""
        text_scorer = MockTextFloatScaleScorer(return_value=0.8)
        audio_scorer = AudioFloatScaleScorer(text_capable_scorer=text_scorer)

        # Mock the transcription to return empty string
        with patch.object(
            audio_scorer._audio_helper, "_transcribe_audio_async", new_callable=AsyncMock
        ) as mock_transcribe:
            mock_transcribe.return_value = ""

            scores = await audio_scorer._score_piece_async(audio_message_piece)

            # Empty transcript returns empty list
            assert len(scores) == 0
