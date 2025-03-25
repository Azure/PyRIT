# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import pytest

from pyrit.prompt_converter import ConverterResult
from pyrit.prompt_converter.ascii_smuggler_converter import AsciiSmugglerConverter


@pytest.mark.asyncio
async def test_convert_async_encode_unicode_tags():
    # Test encoding using the Unicode Tags mode with control tags enabled.
    converter = AsciiSmugglerConverter(action="encode", unicode_tags=True, encoding_mode="unicode_tags")
    prompt = "Hello, World!"
    result = await converter.convert_async(prompt=prompt, input_type="text")
    assert isinstance(result, ConverterResult)
    assert result.output_type == "text"
    # In Unicode Tags mode with unicode_tags flag True, the output should start with U+E0001 and end with U+E007F.
    assert result.output_text.startswith(chr(0xE0001))
    assert result.output_text.endswith(chr(0xE007F))


@pytest.mark.asyncio
async def test_convert_async_decode_unicode_tags():
    # The following encoded message is "Hi"
    encoded_message = chr(0xE0001) + chr(0xE0000) + chr(0xE0048) + chr(0xE0000) + chr(0xE0069) + chr(0xE007F)
    converter = AsciiSmugglerConverter(action="decode", encoding_mode="unicode_tags")
    result = await converter.convert_async(prompt=encoded_message, input_type="text")
    assert isinstance(result, ConverterResult)
    assert result.output_type == "text"
    assert result.output_text == "Hi"


@pytest.mark.asyncio
async def test_encode_decode_unicode_tags():
    # Test round-trip encoding/decoding with Unicode Tags mode (with control tags).
    base_string = "Hello, World!"
    encode_converter = AsciiSmugglerConverter(action="encode", unicode_tags=True, encoding_mode="unicode_tags")
    encoded_result = await encode_converter.convert_async(prompt=base_string, input_type="text")

    decode_converter = AsciiSmugglerConverter(action="decode", encoding_mode="unicode_tags")
    decoded_result = await decode_converter.convert_async(prompt=encoded_result.output_text, input_type="text")

    assert isinstance(decoded_result, ConverterResult)
    assert decoded_result.output_type == "text"
    assert decoded_result.output_text == base_string


@pytest.mark.asyncio
async def test_encode_decode_unicode_tags_notags():
    # Test encoding and decoding without unicode tags to ensure input integrity.
    base_string = "Hello, World!"
    encode_converter = AsciiSmugglerConverter(action="encode", unicode_tags=False, encoding_mode="unicode_tags")
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


# Test for the input_supported method
def test_input_supported():
    converter = AsciiSmugglerConverter()
    assert converter.input_supported("text") is True
    assert converter.input_supported("image") is False
    assert converter.input_supported("audio") is False
    assert converter.input_supported("video") is False
    assert converter.input_supported("other") is False
