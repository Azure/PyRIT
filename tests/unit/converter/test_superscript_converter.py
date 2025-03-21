# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import pytest
import random

from pyrit.prompt_converter import ConverterResult, SuperscriptConverter


async def _check_conversion(converter, prompts, expected_outputs):
    for prompt, expected_output in zip(prompts, expected_outputs):
        result = await converter.convert_async(prompt=prompt, input_type="text")
        assert isinstance(result, ConverterResult)
        assert result.output_text == expected_output


@pytest.mark.asyncio
async def test_superscript_converter():
    defalut_converter = SuperscriptConverter()
    await _check_conversion(
        defalut_converter,
        ["Let's test this converter!", "Unsupported characters stay the same: qCFQSXYZ"],
        ["ᴸᵉᵗ'ˢ ᵗᵉˢᵗ ᵗʰⁱˢ ᶜᵒⁿᵛᵉʳᵗᵉʳ!", "ᵁⁿˢᵘᵖᵖᵒʳᵗᵉᵈ ᶜʰᵃʳᵃᶜᵗᵉʳˢ ˢᵗᵃʸ ᵗʰᵉ ˢᵃᵐᵉ: qCFQSXYZ"],
    )

    alternate_converter = SuperscriptConverter(mode="alternate")
    await _check_conversion(
        alternate_converter,
        ["word1 word2 word3 word4 word5"],
        ["ʷᵒʳᵈ¹ word2 ʷᵒʳᵈ³ word4 ʷᵒʳᵈ⁵"],
    )

@pytest.mark.asyncio
async def test_random_superscript_converter():
    full_random_converter = SuperscriptConverter(mode="random", random_percentage=100)
    await _check_conversion(
        full_random_converter,
        ["Let's test random mode"],
        ["ᴸᵉᵗ'ˢ ᵗᵉˢᵗ ʳᵃⁿᵈᵒᵐ ᵐᵒᵈᵉ"],
    )
    zero_random_converter = SuperscriptConverter(mode="random", random_percentage=0)
    await _check_conversion(
        zero_random_converter,
        ["Let's test random mode"],
        ["Let's test random mode"],
    )

    random.seed(32)  # with seed=32 and 6 words, words at [1,2,5] will be converted
    half_random_converter = SuperscriptConverter(mode="random", random_percentage=50)
    test_text = "one two three four five six"
    expected_output = "ᵒⁿᵉ ᵗʷᵒ three four ᶠⁱᵛᵉ six"
    result = await half_random_converter.convert_async(prompt=test_text, input_type="text")
    assert result.output_text == expected_output
