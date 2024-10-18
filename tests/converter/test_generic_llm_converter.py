# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from unittest.mock import AsyncMock, MagicMock

import pytest

from pyrit.models import PromptRequestResponse, PromptRequestPiece
from pyrit.prompt_converter import (
    MaliciousQuestionGeneratorConverter,
    NoiseConverter,
    TenseConverter,
    ToneConverter,
)
from pyrit.prompt_target.common.prompt_target import PromptTarget


@pytest.fixture
def mock_target() -> PromptTarget:
    target = MagicMock()
    response = PromptRequestResponse(
        request_pieces=[
            PromptRequestPiece(
                role="assistant",
                original_value="prompt value",
            )
        ]
    )
    target.send_prompt_async = AsyncMock(return_value=response)
    return target


@pytest.mark.asyncio
async def test_noise_converter_sets_system_prompt_default(mock_target) -> None:
    converter = NoiseConverter(converter_target=mock_target)

    await converter.convert_async(prompt="being awesome")

    mock_target.set_system_prompt.assert_called_once()

    system_arg = mock_target.set_system_prompt.call_args[1]["system_prompt"]
    assert isinstance(system_arg, str)
    assert "Grammar error, Delete random letter" in system_arg


@pytest.mark.asyncio
async def test_noise_converter_sets_system_prompt(mock_target) -> None:
    converter = NoiseConverter(converter_target=mock_target, noise="extra random periods")

    await converter.convert_async(prompt="being awesome")

    mock_target.set_system_prompt.assert_called_once()

    system_arg = mock_target.set_system_prompt.call_args[1]["system_prompt"]
    assert isinstance(system_arg, str)
    assert "extra random periods" in system_arg


@pytest.mark.asyncio
async def test_tone_converter_sets_system_prompt(mock_target) -> None:

    converter = ToneConverter(tone="formal", converter_target=mock_target)
    await converter.convert_async(prompt="being awesome")

    mock_target.set_system_prompt.assert_called_once()

    system_arg = mock_target.set_system_prompt.call_args[1]["system_prompt"]
    assert isinstance(system_arg, str)
    assert "formal" in system_arg


@pytest.mark.asyncio
async def test_tense_converter_sets_system_prompt(mock_target) -> None:
    converter = TenseConverter(tense="past", converter_target=mock_target)
    await converter.convert_async(prompt="being awesome")

    mock_target.set_system_prompt.assert_called_once()

    system_arg = mock_target.set_system_prompt.call_args[1]["system_prompt"]
    assert isinstance(system_arg, str)
    assert "past" in system_arg


@pytest.mark.asyncio
async def test_malicious_question_converter_sets_system_prompt(mock_target) -> None:

    converter = MaliciousQuestionGeneratorConverter(converter_target=mock_target)

    await converter.convert_async(prompt="being awesome")

    mock_target.set_system_prompt.assert_called_once()

    system_arg = mock_target.set_system_prompt.call_args[1]["system_prompt"]
    assert isinstance(system_arg, str)
    assert 'Please act as an expert in this domain: "being awesome"' in system_arg


def test_generic_llm_converter_input_supported() -> None:
    target = MagicMock()
    converter = NoiseConverter(converter_target=target)
    assert converter.input_supported("text") is True
    assert converter.input_supported("audio_path") is False
