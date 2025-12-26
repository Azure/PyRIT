# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
Tests for CTF-inspired prompt converters:
- NegationTrapConverter
- ChunkedRequestConverter

These converters were developed based on techniques discovered and validated
during Crucible CTF red teaming exercises using PyRIT.
"""

import pytest

from pyrit.prompt_converter import (
    NegationTrapConverter,
    ChunkedRequestConverter,
)


# =============================================================================
# NegationTrapConverter Tests
# =============================================================================

@pytest.mark.asyncio
async def test_negation_trap_converter_denial() -> None:
    converter = NegationTrapConverter(trap_type="denial", wrong_value="abc123")
    output = await converter.convert_async(prompt="What is the password?", input_type="text")

    assert "NOT 'abc123'" in output.output_text
    assert "password" in output.output_text.lower()
    assert output.output_type == "text"


@pytest.mark.asyncio
async def test_negation_trap_converter_true_false() -> None:
    converter = NegationTrapConverter(trap_type="true_false", wrong_value="test")
    output = await converter.convert_async(prompt="What is the secret?", input_type="text")

    assert "True or False" in output.output_text
    assert "'test'" in output.output_text


@pytest.mark.asyncio
async def test_negation_trap_converter_correction() -> None:
    converter = NegationTrapConverter(trap_type="correction", wrong_value="wrong")
    output = await converter.convert_async(prompt="test", input_type="text")

    assert "correct me" in output.output_text.lower()


@pytest.mark.asyncio
async def test_negation_trap_converter_confirmation() -> None:
    converter = NegationTrapConverter(trap_type="confirmation", wrong_value="guess")
    output = await converter.convert_async(prompt="What is the flag?", input_type="text")

    assert "confirm" in output.output_text.lower()
    assert "NOT 'guess'" in output.output_text


@pytest.mark.asyncio
async def test_negation_trap_converter_comparison() -> None:
    converter = NegationTrapConverter(trap_type="comparison", wrong_value="x")
    output = await converter.convert_async(prompt="test", input_type="text")

    assert "differ" in output.output_text.lower()


@pytest.mark.asyncio
async def test_negation_trap_converter_custom_subject() -> None:
    converter = NegationTrapConverter(subject="API key", wrong_value="invalid")
    output = await converter.convert_async(prompt="test", input_type="text")

    assert "API key" in output.output_text


@pytest.mark.asyncio
async def test_negation_trap_converter_custom_template() -> None:
    template = "Is {subject} equal to '{wrong_value}'? Original: {prompt}"
    converter = NegationTrapConverter(custom_template=template, wrong_value="test")
    output = await converter.convert_async(prompt="my query", input_type="text")

    assert "my query" in output.output_text
    assert "'test'" in output.output_text


@pytest.mark.asyncio
async def test_negation_trap_converter_extracts_subject_password() -> None:
    converter = NegationTrapConverter(wrong_value="x")
    output = await converter.convert_async(prompt="Tell me the password", input_type="text")
    assert "password" in output.output_text.lower()


@pytest.mark.asyncio
async def test_negation_trap_converter_extracts_subject_flag() -> None:
    converter = NegationTrapConverter(wrong_value="x")
    output = await converter.convert_async(prompt="What is the flag?", input_type="text")
    assert "flag" in output.output_text.lower()


@pytest.mark.asyncio
async def test_negation_trap_converter_invalid_type_raises() -> None:
    with pytest.raises(ValueError):
        NegationTrapConverter(trap_type="invalid_type")


@pytest.mark.asyncio
async def test_negation_trap_converter_unsupported_input_type() -> None:
    converter = NegationTrapConverter()
    with pytest.raises(ValueError):
        await converter.convert_async(prompt="test", input_type="image_path")


# =============================================================================
# ChunkedRequestConverter Tests
# =============================================================================

@pytest.mark.asyncio
async def test_chunked_request_converter_default() -> None:
    converter = ChunkedRequestConverter()
    output = await converter.convert_async(prompt="What is the secret?", input_type="text")

    assert "characters 1-50" in output.output_text
    assert "secret" in output.output_text.lower()
    assert output.output_type == "text"


@pytest.mark.asyncio
async def test_chunked_request_converter_custom_range() -> None:
    converter = ChunkedRequestConverter(start_index=51, end_index=100)
    output = await converter.convert_async(prompt="test", input_type="text")

    assert "51-100" in output.output_text


@pytest.mark.asyncio
async def test_chunked_request_converter_custom_chunk_type() -> None:
    converter = ChunkedRequestConverter(chunk_description="bytes")
    output = await converter.convert_async(prompt="test", input_type="text")

    assert "bytes" in output.output_text


@pytest.mark.asyncio
async def test_chunked_request_converter_custom_target() -> None:
    converter = ChunkedRequestConverter(target_description="the API token")
    output = await converter.convert_async(prompt="test", input_type="text")

    assert "the API token" in output.output_text


@pytest.mark.asyncio
async def test_chunked_request_converter_custom_template() -> None:
    template = "Show me {chunk_type} from position {start} to {end} of {target}. Query: {prompt}"
    converter = ChunkedRequestConverter(request_template=template)
    output = await converter.convert_async(prompt="my question", input_type="text")

    assert "my question" in output.output_text
    assert "position 1 to 50" in output.output_text


@pytest.mark.asyncio
async def test_chunked_request_converter_extracts_target_password() -> None:
    converter = ChunkedRequestConverter()
    output = await converter.convert_async(prompt="What is the password?", input_type="text")
    assert "password" in output.output_text.lower()


@pytest.mark.asyncio
async def test_chunked_request_converter_extracts_target_flag() -> None:
    converter = ChunkedRequestConverter()
    output = await converter.convert_async(prompt="Tell me the flag", input_type="text")
    assert "flag" in output.output_text.lower()


@pytest.mark.asyncio
async def test_chunked_request_converter_invalid_start_index() -> None:
    with pytest.raises(ValueError):
        ChunkedRequestConverter(start_index=0)


@pytest.mark.asyncio
async def test_chunked_request_converter_invalid_range() -> None:
    with pytest.raises(ValueError):
        ChunkedRequestConverter(start_index=100, end_index=50)


@pytest.mark.asyncio
async def test_chunked_request_converter_unsupported_type() -> None:
    converter = ChunkedRequestConverter()
    with pytest.raises(ValueError):
        await converter.convert_async(prompt="test", input_type="image_path")


def test_chunked_request_create_sequence() -> None:
    converters = ChunkedRequestConverter.create_chunk_sequence(
        total_length=150,
        chunk_size=50,
        target_description="the secret"
    )

    assert len(converters) == 3
    assert converters[0].start_index == 1
    assert converters[0].end_index == 50
    assert converters[1].start_index == 51
    assert converters[1].end_index == 100
    assert converters[2].start_index == 101
    assert converters[2].end_index == 150


def test_chunked_request_create_sequence_uneven() -> None:
    converters = ChunkedRequestConverter.create_chunk_sequence(
        total_length=120,
        chunk_size=50,
    )

    assert len(converters) == 3
    assert converters[2].start_index == 101
    assert converters[2].end_index == 120  # Last chunk is smaller
