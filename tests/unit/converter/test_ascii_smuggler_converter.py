# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import pytest

from pyrit.prompt_converter import ConverterResult
from pyrit.prompt_converter.ascii_smuggler_converter import AsciiSmugglerConverter


@pytest.mark.asyncio
async def test_convert_async_encode_unicode_tags_control():
    # Test encoding using the Unicode Tags mode with control tags enabled.
    converter = AsciiSmugglerConverter(action="encode", encoding_mode="unicode_tags_control")
    prompt = "Hello, World!"
    result = await converter.convert_async(prompt=prompt, input_type="text")
    assert isinstance(result, ConverterResult)
    assert result.output_type == "text"
    # In Unicode Tags mode with unicode_tags flag True, the output should start with U+E0001 and end with U+E007F.
    assert result.output_text.startswith(chr(0xE0001))
    assert result.output_text.endswith(chr(0xE007F))


@pytest.mark.asyncio
async def test_convert_async_encode_unicode_tags():
    # Test encoding using the Unicode Tags mode without control tags.
    converter = AsciiSmugglerConverter(action="encode", encoding_mode="unicode_tags")
    prompt = "Hello, World!"
    result = await converter.convert_async(prompt=prompt, input_type="text")
    assert isinstance(result, ConverterResult)
    assert result.output_type == "text"
    assert not result.output_text.startswith(chr(0xE0001))
    assert not result.output_text.endswith(chr(0xE007F))


@pytest.mark.asyncio
async def test_convert_async_decode_unicode_tags():
    # The following encoded message is "Hi" with control tags
    encoded_message = chr(0xE0001) + chr(0xE0000) + chr(0xE0048) + chr(0xE0000) + chr(0xE0069) + chr(0xE007F)
    converter = AsciiSmugglerConverter(action="decode", encoding_mode="unicode_tags_control")
    result = await converter.convert_async(prompt=encoded_message, input_type="text")
    assert isinstance(result, ConverterResult)
    assert result.output_type == "text"
    assert result.output_text == "Hi"


@pytest.mark.asyncio
async def test_encode_decode_unicode_tags_control():
    # Test round-trip encoding/decoding with Unicode Tags control mode.
    base_string = "Hello, World!"
    encode_converter = AsciiSmugglerConverter(action="encode", encoding_mode="unicode_tags_control")
    encoded_result = await encode_converter.convert_async(prompt=base_string, input_type="text")

    decode_converter = AsciiSmugglerConverter(action="decode", encoding_mode="unicode_tags_control")
    decoded_result = await decode_converter.convert_async(prompt=encoded_result.output_text, input_type="text")

    assert isinstance(decoded_result, ConverterResult)
    assert decoded_result.output_type == "text"
    assert decoded_result.output_text == base_string


@pytest.mark.asyncio
async def test_encode_decode_unicode_tags():
    # Test encoding and decoding without unicode tags to ensure input integrity.
    base_string = "Hello, World!"
    encode_converter = AsciiSmugglerConverter(action="encode", encoding_mode="unicode_tags")
    encoded_result = await encode_converter.convert_async(prompt=base_string, input_type="text")

    decode_converter = AsciiSmugglerConverter(action="decode", encoding_mode="unicode_tags")
    decoded_result = await decode_converter.convert_async(prompt=encoded_result.output_text, input_type="text")

    assert isinstance(decoded_result, ConverterResult)
    assert decoded_result.output_type == "text"
    assert decoded_result.output_text == base_string


@pytest.mark.asyncio
async def test_convert_async_encode_sneaky_bits():
    # Test encoding using the Sneaky Bits mode.
    converter = AsciiSmugglerConverter(action="encode", encoding_mode="sneaky_bits")
    prompt = "Hello, World!"
    result = await converter.convert_async(prompt=prompt, input_type="text")
    assert isinstance(result, ConverterResult)
    assert result.output_type == "text"
    # In Sneaky Bits mode, the output should consist solely of the two invisible bit characters.
    valid_chars = {converter.zero_char, converter.one_char}
    assert all(ch in valid_chars for ch in result.output_text)


@pytest.mark.asyncio
async def test_convert_async_decode_sneaky_bits():
    # Test decoding using the Sneaky Bits mode.
    original_text = "Test Sneaky Bits"
    encode_converter = AsciiSmugglerConverter(action="encode", encoding_mode="sneaky_bits")
    encoded_result = await encode_converter.convert_async(prompt=original_text, input_type="text")

    decode_converter = AsciiSmugglerConverter(action="decode", encoding_mode="sneaky_bits")
    decoded_result = await decode_converter.convert_async(prompt=encoded_result.output_text, input_type="text")
    assert isinstance(decoded_result, ConverterResult)
    assert decoded_result.output_type == "text"
    assert decoded_result.output_text == original_text


@pytest.mark.asyncio
async def test_convert_async_encode_only_modes():
    prompt = "Test"

    # Unicode Tags Control
    converter_control = AsciiSmugglerConverter(action="encode", encoding_mode="unicode_tags_control")
    result_control = await converter_control.convert_async(prompt=prompt, input_type="text")
    assert result_control.output_text != ""
    assert result_control.output_text.startswith(chr(0xE0001))
    assert result_control.output_text.endswith(chr(0xE007F))
    assert len(result_control.output_text) == len(prompt) + 2  # Explicit length check

    # Unicode Tags (no control)
    converter_no_control = AsciiSmugglerConverter(action="encode", encoding_mode="unicode_tags")
    result_no_control = await converter_no_control.convert_async(prompt=prompt, input_type="text")
    assert result_no_control.output_text != ""
    assert not result_no_control.output_text.startswith(chr(0xE0001))
    assert not result_no_control.output_text.endswith(chr(0xE007F))
    assert len(result_no_control.output_text) == len(prompt)  # Explicit length check

    # Sneaky Bits
    converter_sneaky = AsciiSmugglerConverter(action="encode", encoding_mode="sneaky_bits")
    result_sneaky = await converter_sneaky.convert_async(prompt=prompt, input_type="text")
    assert result_sneaky.output_text != ""
    valid_chars = {converter_sneaky.zero_char, converter_sneaky.one_char}
    assert all(ch in valid_chars for ch in result_sneaky.output_text)
    assert len(result_sneaky.output_text) == len(prompt.encode("utf-8")) * 8  # Explicit length check


# Test for the input_supported method
def test_input_supported():
    converter = AsciiSmugglerConverter()
    assert converter.input_supported("text") is True
    assert converter.input_supported("image") is False
    assert converter.input_supported("audio") is False
    assert converter.input_supported("video") is False
    assert converter.input_supported("other") is False
