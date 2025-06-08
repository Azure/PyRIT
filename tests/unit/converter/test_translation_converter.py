# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from unittest.mock import AsyncMock, patch

import pytest
from unit.mocks import MockPromptTarget

from pyrit.models import PromptRequestPiece, PromptRequestResponse
from pyrit.prompt_converter import TranslationConverter


def test_prompt_translation_init_templates_not_null(duckdb_instance):
    prompt_target = MockPromptTarget()
    translation_converter = TranslationConverter(converter_target=prompt_target, language="en")
    assert translation_converter.system_prompt


@pytest.mark.parametrize("languages", [None, ""])
def test_translator_converter_languages_validation_throws(languages, duckdb_instance):
    prompt_target = MockPromptTarget()
    with pytest.raises(ValueError):
        TranslationConverter(converter_target=prompt_target, language=languages)


@pytest.mark.asyncio
async def test_translation_converter_convert_async_retrieve_key_capitalization_mismatch(duckdb_instance):
    prompt_target = MockPromptTarget()

    translation_converter = TranslationConverter(converter_target=prompt_target, language="spanish")
    with patch.object(translation_converter, "_send_translation_prompt_async", new=AsyncMock(return_value="hola")):

        raised = False
        try:
            await translation_converter.convert_async(prompt="hello")
        except KeyError:
            raised = True  # There should be no KeyError

        assert raised is False


@pytest.mark.asyncio
async def test_translation_converter_retries_on_exception(duckdb_instance):
    prompt_target = MockPromptTarget()
    max_retries = 3
    translation_converter = TranslationConverter(
        converter_target=prompt_target, language="spanish", max_retries=max_retries
    )

    mock_send_prompt = AsyncMock(side_effect=Exception("Test failure"))
    with patch.object(prompt_target, "send_prompt_async", mock_send_prompt):
        with pytest.raises(Exception):
            await translation_converter.convert_async(prompt="hello")

        assert mock_send_prompt.call_count == max_retries


@pytest.mark.asyncio
async def test_translation_converter_succeeds_after_retries(duckdb_instance):
    """Test that TranslationConverter succeeds if a retry attempt works."""
    prompt_target = MockPromptTarget()
    max_retries = 3
    translation_converter = TranslationConverter(
        converter_target=prompt_target, language="spanish", max_retries=max_retries
    )

    success_response = PromptRequestResponse(
        request_pieces=[
            PromptRequestPiece(
                role="assistant",
                conversation_id="test-id",
                original_value="hello",
                converted_value="hola",
                original_value_data_type="text",
                converted_value_data_type="text",
                prompt_target_identifier={"target": "test-identifier"},
                sequence=1,
            )
        ]
    )

    # fail twice, then succeed
    mock_send_prompt = AsyncMock()
    mock_send_prompt.side_effect = [Exception("First failure"), Exception("Second failure"), success_response]

    with patch.object(prompt_target, "send_prompt_async", mock_send_prompt):
        result = await translation_converter.convert_async(prompt="hello")

        assert mock_send_prompt.call_count == max_retries
        assert result.output_text == "hola"
        assert result.output_type == "text"


def test_translation_converter_input_supported(duckdb_instance):
    prompt_target = MockPromptTarget()
    translation_converter = TranslationConverter(converter_target=prompt_target, language="spanish")
    assert translation_converter.input_supported("text") is True
    assert translation_converter.input_supported("image_path") is False
