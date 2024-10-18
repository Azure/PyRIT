# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import pytest
from pyrit.prompt_converter import ConverterResult
from pyrit.prompt_converter.ascii_smuggler_converter import AsciiSmugglerConverter


@pytest.mark.asyncio
async def test_convert_async_encode():
    converter = AsciiSmugglerConverter(action="encode", unicode_tags=True)
    prompt = "Hello, World!"
    result = await converter.convert_async(prompt=prompt, input_type="text")
    assert isinstance(result, ConverterResult)
    assert result.output_type == "text"
    assert result.output_text.startswith(chr(0xE0001))  # Check for Unicode tag at the start
    assert result.output_text.endswith(chr(0xE007F))  # Check for Unicode tag at the end


@pytest.mark.asyncio
async def test_convert_async_decode():
    # The following encoded message is "Hi"
    encoded_message = chr(0xE0001) + chr(0xE0000) + chr(0xE0048) + chr(0xE0000) + chr(0xE0069) + chr(0xE007F)
    converter = AsciiSmugglerConverter(action="decode")
    result = await converter.convert_async(prompt=encoded_message, input_type="text")
    assert isinstance(result, ConverterResult)
    assert result.output_type == "text"
    assert result.output_text == "Hi"


@pytest.mark.asyncio
async def test_encode_decode():
    base_string = "Hello, World!"
    encode_converter = AsciiSmugglerConverter(action="encode", unicode_tags=True)
    encoded_result = await encode_converter.convert_async(prompt=base_string, input_type="text")

    decode_converter = AsciiSmugglerConverter(action="decode")
    decoded_reults = await decode_converter.convert_async(prompt=encoded_result.output_text, input_type="text")

    assert isinstance(decoded_reults, ConverterResult)
    assert decoded_reults.output_type == "text"
    assert decoded_reults.output_text == base_string


@pytest.mark.asyncio
async def test_encode_decode_notags():
    base_string = "Hello, World!"
    encode_converter = AsciiSmugglerConverter(action="encode", unicode_tags=False)
    encoded_result = await encode_converter.convert_async(prompt=base_string, input_type="text")

    decode_converter = AsciiSmugglerConverter(action="decode")
    decoded_reults = await decode_converter.convert_async(prompt=encoded_result.output_text, input_type="text")

    assert isinstance(decoded_reults, ConverterResult)
    assert decoded_reults.output_type == "text"
    assert decoded_reults.output_text == base_string


# Test for the input_supported method
def test_input_supported():
    converter = AsciiSmugglerConverter()
    assert converter.input_supported("text") is True
    assert converter.input_supported("image") is False
    assert converter.input_supported("audio") is False
    assert converter.input_supported("video") is False
    assert converter.input_supported("other") is False
