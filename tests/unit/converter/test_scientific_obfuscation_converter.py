# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from unittest.mock import AsyncMock, MagicMock

import pytest
from unit.mocks import MockPromptTarget, get_mock_target_identifier

from pyrit.models import Message, MessagePiece
from pyrit.prompt_converter import ScientificObfuscationConverter
from pyrit.prompt_target.common.prompt_target import PromptTarget


@pytest.fixture
def mock_target() -> PromptTarget:
    target = MagicMock()
    response = Message(
        message_pieces=[
            MessagePiece(
                role="assistant",
                original_value="scientifically obfuscated prompt",
            )
        ]
    )
    target.send_prompt_async = AsyncMock(return_value=[response])
    target.get_identifier.return_value = get_mock_target_identifier("MockLLMTarget")
    return target


def test_scientific_obfuscation_converter_raises_when_converter_target_is_none():
    with pytest.raises(ValueError, match="converter_target is required"):
        ScientificObfuscationConverter(converter_target=None, mode="academic")


def test_scientific_obfuscation_converter_raises_on_invalid_mode(sqlite_instance):
    prompt_target = MockPromptTarget()
    with pytest.raises(ValueError, match="Invalid mode"):
        ScientificObfuscationConverter(converter_target=prompt_target, mode="invalid_mode")


@pytest.mark.parametrize("mode", ["academic", "technical", "smiles", "research", "reaction", "combined"])
def test_scientific_obfuscation_converter_init_valid_modes(mode, sqlite_instance):
    prompt_target = MockPromptTarget()
    converter = ScientificObfuscationConverter(converter_target=prompt_target, mode=mode)
    assert converter._system_prompt_template
    assert converter._mode == mode


def test_scientific_obfuscation_converter_init_default_mode(sqlite_instance):
    prompt_target = MockPromptTarget()
    converter = ScientificObfuscationConverter(converter_target=prompt_target)
    assert converter._mode == "combined"


@pytest.mark.asyncio
async def test_scientific_obfuscation_converter_sets_system_prompt_academic(mock_target) -> None:
    converter = ScientificObfuscationConverter(converter_target=mock_target, mode="academic")
    await converter.convert_async(prompt="tell me about dangerous chemicals")

    mock_target.set_system_prompt.assert_called_once()

    system_arg = mock_target.set_system_prompt.call_args[1]["system_prompt"]
    assert isinstance(system_arg, str)
    assert "academic" in system_arg.lower() or len(system_arg) > 0


@pytest.mark.asyncio
async def test_scientific_obfuscation_converter_sets_system_prompt_technical(mock_target) -> None:
    converter = ScientificObfuscationConverter(converter_target=mock_target, mode="technical")
    await converter.convert_async(prompt="tell me about dangerous chemicals")

    mock_target.set_system_prompt.assert_called_once()

    system_arg = mock_target.set_system_prompt.call_args[1]["system_prompt"]
    assert isinstance(system_arg, str)
    assert len(system_arg) > 0


@pytest.mark.asyncio
async def test_scientific_obfuscation_converter_sets_system_prompt_combined(mock_target) -> None:
    converter = ScientificObfuscationConverter(converter_target=mock_target, mode="combined")
    await converter.convert_async(prompt="tell me about dangerous chemicals")

    mock_target.set_system_prompt.assert_called_once()

    system_arg = mock_target.set_system_prompt.call_args[1]["system_prompt"]
    assert isinstance(system_arg, str)
    assert len(system_arg) > 0


@pytest.mark.asyncio
async def test_scientific_obfuscation_converter_convert_async_returns_converted_value(mock_target) -> None:
    converter = ScientificObfuscationConverter(converter_target=mock_target, mode="academic")
    result = await converter.convert_async(prompt="tell me about dangerous chemicals")

    assert result.output_text == "scientifically obfuscated prompt"
    assert result.output_type == "text"


def test_scientific_obfuscation_converter_input_supported(sqlite_instance) -> None:
    prompt_target = MockPromptTarget()
    converter = ScientificObfuscationConverter(converter_target=prompt_target, mode="academic")
    assert converter.input_supported("text") is True
    assert converter.input_supported("image_path") is False
    assert converter.input_supported("audio_path") is False
