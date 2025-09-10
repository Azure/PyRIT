# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import pytest
from datetime import datetime
from unittest.mock import MagicMock, patch, AsyncMock

from pyrit.executor.attack.printer.markdown_printer import MarkdownAttackResultPrinter
from pyrit.memory import CentralMemory
from pyrit.models import AttackResult, Score, AttackOutcome
from pyrit.models.prompt_request_piece import PromptRequestPiece
from pyrit.models.prompt_request_response import PromptRequestResponse


@pytest.fixture
def mock_memory():
    memory = MagicMock(spec=CentralMemory)
    with patch('pyrit.executor.attack.printer.markdown_printer.CentralMemory') as mock_central_memory:
        mock_central_memory.get_memory_instance.return_value = memory
        yield memory


@pytest.fixture
def markdown_printer():
    return MarkdownAttackResultPrinter(display_inline=False)


@pytest.fixture
def sample_score():
    return Score(
        score_type="test",
        score_value=0.85,
        score_category="accuracy",
        score_rationale="Test rationale",
        score_metadata={"key": "value"}
    )


@pytest.fixture
def sample_attack_result():
    return AttackResult(
        objective="Test objective",
        attack_identifier={"__type__": "TestAttack"},
        conversation_id="test-conv-123",
        executed_turns=3,
        execution_time_ms=1500,
        outcome=AttackOutcome.SUCCESS,
        outcome_reason="Test successful",
        last_score=Score(
            score_type="final",
            score_value=0.9,
            score_category="overall",
            score_rationale="Final test score"
        )
    )


@pytest.fixture
def sample_prompt_piece():
    return PromptRequestPiece(
        role="user",
        original_value="Original text",
        converted_value="Converted text",
        converted_value_data_type="text"
    )


@pytest.fixture
def sample_prompt_response(sample_prompt_piece):
    response = PromptRequestResponse()
    response.add_piece(sample_prompt_piece)
    return response


def test_init(mock_memory):
    """Test MarkdownAttackResultPrinter initialization."""
    printer = MarkdownAttackResultPrinter(display_inline=True)
    assert printer._display_inline is True
    assert printer._memory is mock_memory


def test_format_score_float(markdown_printer, sample_score):
    """Test score formatting with float value."""
    formatted = markdown_printer._format_score(sample_score)
    assert "**0.85**" in formatted
    assert "**Score Type:** test" in formatted
    assert "**Category:** accuracy" in formatted
    assert "**Rationale:** Test rationale" in formatted
    assert "**Metadata:** `{'key': 'value'}`" in formatted


def test_format_score_bool(markdown_printer):
    """Test score formatting with boolean value."""
    score = Score(score_type="test", score_value=True)
    formatted = markdown_printer._format_score(score)
    assert "True" in formatted
    assert "**Score Type:** test" in formatted
    assert "**Category:** N/A" in formatted


def test_format_score_multiline_rationale(markdown_printer):
    """Test score formatting with multi-line rationale."""
    score = Score(
        score_type="test",
        score_value=1,
        score_rationale="Line 1\nLine 2\nLine 3"
    )
    formatted = markdown_printer._format_score(score)
    assert "Line 1" in formatted
    assert "Line 2" in formatted
    assert "Line 3" in formatted


def test_get_audio_mime_type(markdown_printer):
    """Test audio MIME type detection."""
    assert markdown_printer._get_audio_mime_type(audio_path="test.wav") == "audio/wav"
    assert markdown_printer._get_audio_mime_type(audio_path="test.ogg") == "audio/ogg"
    assert markdown_printer._get_audio_mime_type(audio_path="test.m4a") == "audio/mp4"
    assert markdown_printer._get_audio_mime_type(audio_path="test.mp3") == "audio/mpeg"


def test_format_image_content(markdown_printer):
    """Test image content formatting."""
    image_path = os.path.join("test", "path", "image.png")
    formatted = markdown_printer._format_image_content(image_path=image_path)
    assert formatted[0].startswith("![Image]")
    assert "image.png" in formatted[0]


def test_format_audio_content(markdown_printer):
    """Test audio content formatting."""
    audio_path = "test.wav"
    formatted = markdown_printer._format_audio_content(audio_path=audio_path)
    assert "<audio controls>" in formatted
    assert 'type="audio/wav"' in "\n".join(formatted)
    assert "Your browser does not support the audio element." in formatted


def test_format_error_content(markdown_printer, sample_prompt_piece):
    """Test error content formatting."""
    sample_prompt_piece.response_error = "TestError"
    formatted = markdown_printer._format_error_content(piece=sample_prompt_piece)
    assert "**Error Response:**" in formatted
    assert "*Error Type: TestError*" in formatted
    assert "```json" in formatted


def test_format_text_content_with_conversion(markdown_printer, sample_prompt_piece):
    """Test text content formatting when original and converted values differ."""
    formatted = markdown_printer._format_text_content(piece=sample_prompt_piece, show_original=True)
    assert "**Original:**" in formatted
    assert "Original text" in formatted
    assert "**Converted:**" in formatted
    assert "Converted text" in formatted


def test_format_text_content_without_conversion(markdown_printer, sample_prompt_piece):
    """Test text content formatting when values are the same."""
    sample_prompt_piece.converted_value = sample_prompt_piece.original_value
    formatted = markdown_printer._format_text_content(piece=sample_prompt_piece, show_original=True)
    assert "**Original:**" not in formatted
    assert sample_prompt_piece.original_value in formatted


@pytest.mark.asyncio
async def test_format_piece_content_image(markdown_printer, sample_prompt_piece):
    """Test piece content formatting for images."""
    sample_prompt_piece.converted_value_data_type = "image_path"
    sample_prompt_piece.converted_value = "test.png"
    formatted = await markdown_printer._format_piece_content_async(piece=sample_prompt_piece, show_original=False)
    assert any("![Image]" in line for line in formatted)


@pytest.mark.asyncio
async def test_format_piece_content_audio(markdown_printer, sample_prompt_piece):
    """Test piece content formatting for audio."""
    sample_prompt_piece.converted_value_data_type = "audio_path"
    sample_prompt_piece.converted_value = "test.wav"
    formatted = await markdown_printer._format_piece_content_async(piece=sample_prompt_piece, show_original=False)
    assert any("<audio controls>" in line for line in formatted)


@pytest.mark.asyncio
async def test_format_piece_content_error(markdown_printer, sample_prompt_piece):
    """Test piece content formatting for errors."""
    sample_prompt_piece.response_error = "TestError"
    formatted = await markdown_printer._format_piece_content_async(piece=sample_prompt_piece, show_original=False)
    assert any("**Error Response:**" in line for line in formatted)


@pytest.mark.asyncio
async def test_print_result_async(markdown_printer, sample_attack_result, mock_memory, capsys):
    """Test full attack result printing."""
    mock_memory.get_conversation.return_value = []
    
    await markdown_printer.print_result_async(sample_attack_result)
    captured = capsys.readouterr()
    
    # Check for main sections
    assert "# 🎯 Attack Result: SUCCESS" in captured.out
    assert "## Attack Summary" in captured.out
    assert "### Basic Information" in captured.out
    assert "### Execution Metrics" in captured.out
    assert "### Outcome" in captured.out
    assert "### Final Score" in captured.out
    assert "## Conversation History" in captured.out


@pytest.mark.asyncio
async def test_print_conversation_async(markdown_printer, sample_attack_result, mock_memory, capsys):
    """Test conversation history printing."""
    mock_memory.get_conversation.return_value = []
    
    await markdown_printer.print_conversation_async(sample_attack_result)
    captured = capsys.readouterr()
    
    assert "*No conversation found for ID: test-conv-123*" in captured.out


@pytest.mark.asyncio
async def test_print_summary_async(markdown_printer, sample_attack_result, capsys):
    """Test attack summary printing."""
    await markdown_printer.print_summary_async(sample_attack_result)
    captured = capsys.readouterr()
    
    assert "## Attack Summary" in captured.out
    assert "Test objective" in captured.out
    assert "TestAttack" in captured.out
    assert "test-conv-123" in captured.out
