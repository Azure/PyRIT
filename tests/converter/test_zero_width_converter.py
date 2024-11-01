# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import pytest
from pyrit.prompt_converter import ConverterResult
from pyrit.prompt_converter.zero_width_converter import ZeroWidthConverter


@pytest.mark.asyncio
async def test_convert_async_injects_zero_width_spaces():
    converter = ZeroWidthConverter()
    text = "Hello"
    expected_output = "H\u200Be\u200Bl\u200Bl\u200Bo"  # Zero-width spaces between each character
    result = await converter.convert_async(prompt=text)
    assert isinstance(result, ConverterResult)
    assert result.output_text == expected_output  # Check if output matches expected result with zero-width spaces


@pytest.mark.asyncio
async def test_convert_async_long_text():
    converter = ZeroWidthConverter()
    text = "This is a longer text used to test the ZeroWidthConverter."
    # Expected output has a zero-width space between every character in `text`
    expected_output = "\u200B".join(text)

    result = await converter.convert_async(prompt=text)

    assert result.output_text == expected_output
    # Verify that the output length is as expected: original length + (number of characters - 1) for zero-width spaces
    assert len(result.output_text) == len(text) + len(text) - 1


@pytest.mark.asyncio
async def test_convert_async_handles_empty_string():
    converter = ZeroWidthConverter()
    text = ""
    result = await converter.convert_async(prompt=text)
    assert result.output_text == ""  # Output should be empty if input is empty


@pytest.mark.asyncio
async def test_convert_async_non_text_input_type():
    converter = ZeroWidthConverter()
    with pytest.raises(ValueError) as excinfo:
        await converter.convert_async(prompt="Hello", input_type="non-text")
    assert "Only 'text' input type is supported" in str(excinfo.value)


def test_input_supported_text():
    converter = ZeroWidthConverter()
    assert converter.input_supported("text") is True  # Should support 'text' input type


def test_input_supported_non_text():
    converter = ZeroWidthConverter()
    assert converter.input_supported("non-text") is False  # Should not support non-'text' input types
