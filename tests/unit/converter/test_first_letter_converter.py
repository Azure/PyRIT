# -*- coding: utf-8 -*-
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import pytest

from pyrit.prompt_converter import FirstLetterConverter


# Test that the default converter settings produce the expected result
@pytest.mark.asyncio
async def test_first_letter_converter_default():
    converter = FirstLetterConverter()
    prompt = "Lorem ipsum dolor sit amet"
    output = await converter.convert_async(prompt=prompt, input_type="text")
    assert output.output_text == "Lidsa"
    assert output.output_type == "text"


# Test that the converter produces the expected result with French punctuation
@pytest.mark.asyncio
async def test_first_letter_converter_french():
    converter = FirstLetterConverter()
    prompt = """
    En 1815, M. Charles-François-Bienvenu Myriel était évêque de Digne.
    C'était un vieillard d'environ soixante-quinze ans ; il occupait le siége de Digne depuis 1806.
    """
    output = await converter.convert_async(prompt=prompt, input_type="text")
    assert output.output_text == "E1MCMéédDCuvdsaiolsdDd1"
    assert output.output_type == "text"


# Test that the converter produces the expected result with non-Latin alphabets
@pytest.mark.asyncio
async def test_first_letter_converter_japanese():
    converter = FirstLetterConverter()
    prompt = """
    ふるいけ や
    かわず とびこむ
    みず の おと
    """
    output = await converter.convert_async(prompt=prompt, input_type="text")
    assert output.output_text == "ふやかとみのお"
    assert output.output_type == "text"
