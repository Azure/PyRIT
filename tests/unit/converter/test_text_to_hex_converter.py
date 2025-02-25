# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import pytest

from pyrit.prompt_converter import ConverterResult, TextToHexConverter


@pytest.mark.asyncio
async def test_text_to_hex_converter_ascii():
    converter = TextToHexConverter()
    prompt = "Test random string[#$!; > 18% \n"  # String of ascii characters
    expected_output = "546573742072616E646F6D20737472696E675B2324213B203E20313825200A"  # hex representation of prompt
    result = await converter.convert_async(prompt=prompt, input_type="text")
    assert isinstance(result, ConverterResult)
    assert result.output_text == expected_output
    assert result.output_type == "text"


@pytest.mark.asyncio
async def test_text_to_hex_converter_extended_ascii():
    converter = TextToHexConverter()
    prompt = "éÄ§æ"  # String of extended ascii characters
    expected_output = "C3A9C384C2A7C3A6"  # hex representation of extended ascii characters
    result = await converter.convert_async(prompt=prompt, input_type="text")
    assert isinstance(result, ConverterResult)
    assert result.output_text == expected_output
    assert result.output_type == "text"


@pytest.mark.asyncio
async def test_text_to_hex_converter_empty_string():
    converter = TextToHexConverter()
    prompt = ""  # Empty input string
    expected_output = ""  # Empty output string
    result = await converter.convert_async(prompt=prompt, input_type="text")
    assert result.output_text == expected_output
    assert result.output_type == "text"


@pytest.mark.asyncio
async def test_text_to_hex_converter_multilingual():
    converter = TextToHexConverter()
    prompt = "বাংলা 日本語 ᬅᬓ᭄ᬱᬭᬩᬮᬶ"  # Bengali, Japanese, Balinese
    expected_output = (
        "E0A6ACE0A6BEE0A682E0A6B2E0A6BE20E697A5E69CACE8AA9E20E1AC85E1AC93E1" "AD84E1ACB1E1ACADE1ACA9E1ACAEE1ACB6"
    )  # hex representation of multilingual string
    result = await converter.convert_async(prompt=prompt, input_type="text")
    assert result.output_text == expected_output
    assert result.output_type == "text"


@pytest.mark.asyncio
async def test_text_to_hex_converter_emoji():
    converter = TextToHexConverter()
    prompt = "😊"  # Emoji character with code point U+1F60A
    expected_output = "F09F988A"  # hex representation of '😊'
    result = await converter.convert_async(prompt=prompt, input_type="text")
    assert isinstance(result, ConverterResult)
    assert result.output_text == expected_output
    assert result.output_type == "text"
