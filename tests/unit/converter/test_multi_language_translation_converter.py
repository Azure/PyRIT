# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from unittest.mock import AsyncMock, patch

import pytest
from unit.mocks import MockPromptTarget

from pyrit.models import Message, MessagePiece
from pyrit.prompt_converter import MultiLanguageTranslationConverter

# ── Initialisation / validation tests ──────────────────────────────────────


def test_raises_when_converter_target_is_none():
    with pytest.raises(ValueError, match="converter_target is required"):
        MultiLanguageTranslationConverter(converter_target=None, languages=["french"])


def test_raises_when_languages_empty(sqlite_instance):
    prompt_target = MockPromptTarget()
    with pytest.raises(ValueError, match="Languages list must be provided"):
        MultiLanguageTranslationConverter(converter_target=prompt_target, languages=[])


@pytest.mark.parametrize("languages", [None, []])
def test_languages_validation_throws(languages, sqlite_instance):
    prompt_target = MockPromptTarget()
    with pytest.raises(ValueError):
        MultiLanguageTranslationConverter(converter_target=prompt_target, languages=languages)


def test_init_sets_languages_lowercase(sqlite_instance):
    prompt_target = MockPromptTarget()
    converter = MultiLanguageTranslationConverter(
        converter_target=prompt_target, languages=["French", "SPANISH", "italian"]
    )
    assert converter.languages == ["french", "spanish", "italian"]


def test_init_system_prompt_template_not_null(sqlite_instance):
    prompt_target = MockPromptTarget()
    converter = MultiLanguageTranslationConverter(converter_target=prompt_target, languages=["french"])
    assert converter._prompt_template is not None


# ── Splitting logic tests ─────────────────────────────────────────────────


def test_split_prompt_even_split():
    """4 words, 2 segments -> 2 words each."""
    segments = MultiLanguageTranslationConverter._split_prompt_into_segments("Hello how are you", 2)
    assert segments == ["Hello how", "are you"]


def test_split_prompt_uneven_split():
    """4 words, 3 segments -> 2-1-1 distribution."""
    segments = MultiLanguageTranslationConverter._split_prompt_into_segments("Hello how are you", 3)
    assert segments == ["Hello how", "are", "you"]


def test_split_prompt_more_languages_than_words():
    """2 words, 5 segments -> only 2 segments (one word each)."""
    segments = MultiLanguageTranslationConverter._split_prompt_into_segments("Hello world", 5)
    assert segments == ["Hello", "world"]


def test_split_prompt_single_word():
    """1 word, 3 segments -> single segment."""
    segments = MultiLanguageTranslationConverter._split_prompt_into_segments("Hello", 3)
    assert segments == ["Hello"]


def test_split_prompt_empty_string():
    """Empty string returns empty list."""
    segments = MultiLanguageTranslationConverter._split_prompt_into_segments("", 3)
    assert segments == []


def test_split_prompt_one_segment():
    """All words in one segment."""
    segments = MultiLanguageTranslationConverter._split_prompt_into_segments("Hello how are you", 1)
    assert segments == ["Hello how are you"]


def test_split_prompt_equal_words_and_segments():
    """Number of segments equals number of words."""
    segments = MultiLanguageTranslationConverter._split_prompt_into_segments("a b c d", 4)
    assert segments == ["a", "b", "c", "d"]


def test_split_prompt_preserves_whitespace_words():
    """Multiple spaces between words are collapsed by split()."""
    segments = MultiLanguageTranslationConverter._split_prompt_into_segments("Hello   how   are   you", 2)
    assert segments == ["Hello how", "are you"]


# ── Async conversion tests ────────────────────────────────────────────────


def _make_response_message(translated_text: str) -> Message:
    """Helper to build a mock response Message."""
    return Message(
        message_pieces=[
            MessagePiece(
                role="assistant",
                conversation_id="test-id",
                original_value="original",
                converted_value=translated_text,
                original_value_data_type="text",
                converted_value_data_type="text",
                prompt_target_identifier={"target": "test-identifier"},
                sequence=1,
            )
        ]
    )


@pytest.mark.asyncio
async def test_convert_async_three_languages(sqlite_instance):
    """Full integration: 4 words, 3 languages -> 3 translated segments joined."""
    prompt_target = MockPromptTarget()
    converter = MultiLanguageTranslationConverter(
        converter_target=prompt_target, languages=["french", "spanish", "italian"]
    )

    responses = [
        [_make_response_message("Bonjour comment")],
        [_make_response_message("estás")],
        [_make_response_message("tu")],
    ]

    mock_send = AsyncMock(side_effect=responses)
    with patch.object(prompt_target, "send_prompt_async", mock_send):
        result = await converter.convert_async(prompt="Hello how are you")

    assert result.output_text == "Bonjour comment estás tu"
    assert result.output_type == "text"
    assert mock_send.call_count == 3


@pytest.mark.asyncio
async def test_convert_async_more_languages_than_words(sqlite_instance):
    """When languages > words, only as many languages as words are used."""
    prompt_target = MockPromptTarget()
    converter = MultiLanguageTranslationConverter(
        converter_target=prompt_target, languages=["french", "spanish", "italian", "german", "japanese"]
    )

    responses = [
        [_make_response_message("Bonjour")],
        [_make_response_message("mundo")],
    ]

    mock_send = AsyncMock(side_effect=responses)
    with patch.object(prompt_target, "send_prompt_async", mock_send):
        result = await converter.convert_async(prompt="Hello world")

    assert result.output_text == "Bonjour mundo"
    assert result.output_type == "text"
    # Only 2 calls, not 5
    assert mock_send.call_count == 2


@pytest.mark.asyncio
async def test_convert_async_single_language(sqlite_instance):
    """Single language behaves like a normal translation."""
    prompt_target = MockPromptTarget()
    converter = MultiLanguageTranslationConverter(converter_target=prompt_target, languages=["french"])

    mock_send = AsyncMock(return_value=[_make_response_message("Bonjour comment allez-vous")])
    with patch.object(prompt_target, "send_prompt_async", mock_send):
        result = await converter.convert_async(prompt="Hello how are you")

    assert result.output_text == "Bonjour comment allez-vous"
    assert result.output_type == "text"
    assert mock_send.call_count == 1


@pytest.mark.asyncio
async def test_convert_async_strips_whitespace_from_response(sqlite_instance):
    """Translated segments should have leading/trailing whitespace stripped."""
    prompt_target = MockPromptTarget()
    converter = MultiLanguageTranslationConverter(converter_target=prompt_target, languages=["french", "spanish"])

    responses = [
        [_make_response_message("  Bonjour  ")],
        [_make_response_message("  mundo  ")],
    ]

    mock_send = AsyncMock(side_effect=responses)
    with patch.object(prompt_target, "send_prompt_async", mock_send):
        result = await converter.convert_async(prompt="Hello world")

    assert result.output_text == "Bonjour mundo"


@pytest.mark.asyncio
async def test_convert_async_unsupported_input_type(sqlite_instance):
    """Passing an unsupported input_type should raise ValueError."""
    prompt_target = MockPromptTarget()
    converter = MultiLanguageTranslationConverter(converter_target=prompt_target, languages=["french"])
    with pytest.raises(ValueError, match="Input type not supported"):
        await converter.convert_async(prompt="Hello", input_type="image_path")


@pytest.mark.asyncio
async def test_convert_async_retries_on_exception(sqlite_instance):
    """Each segment translation should retry on failure up to max_retries."""
    prompt_target = MockPromptTarget()
    max_retries = 3
    converter = MultiLanguageTranslationConverter(
        converter_target=prompt_target, languages=["french"], max_retries=max_retries
    )

    mock_send = AsyncMock(side_effect=Exception("Test failure"))
    with patch.object(prompt_target, "send_prompt_async", mock_send):
        with patch("asyncio.sleep", new_callable=AsyncMock):
            with pytest.raises(Exception):
                await converter.convert_async(prompt="Hello")

            assert mock_send.call_count == max_retries


@pytest.mark.asyncio
async def test_convert_async_succeeds_after_retries(sqlite_instance):
    """Translation should succeed if a retry attempt works."""
    prompt_target = MockPromptTarget()
    converter = MultiLanguageTranslationConverter(converter_target=prompt_target, languages=["french"], max_retries=3)

    mock_send = AsyncMock(
        side_effect=[
            Exception("First failure"),
            Exception("Second failure"),
            [_make_response_message("Bonjour")],
        ]
    )

    with patch.object(prompt_target, "send_prompt_async", mock_send):
        with patch("asyncio.sleep", new_callable=AsyncMock):
            result = await converter.convert_async(prompt="Hello")

    assert result.output_text == "Bonjour"
    assert mock_send.call_count == 3


def test_input_supported(sqlite_instance):
    prompt_target = MockPromptTarget()
    converter = MultiLanguageTranslationConverter(converter_target=prompt_target, languages=["french"])
    assert converter.input_supported("text") is True
    assert converter.input_supported("image_path") is False
