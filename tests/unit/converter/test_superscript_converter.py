# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import math
import random

import pytest

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

    # Test with a longer text (37 words) and 20% conversion rate

    random.seed()
    twenty_percent_converter = SuperscriptConverter(mode="random", random_percentage=20)

    long_text = "Prompt: Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat."
    word_count = len(long_text.split())
    assert word_count == 37

    result = await twenty_percent_converter.convert_async(prompt=long_text, input_type="text")
    original_words = long_text.split()
    converted_words = result.output_text.split()
    assert len(converted_words) == len(original_words)

    # Count words that were actually converted
    converted_count = sum(1 for original, converted in zip(original_words, converted_words) if original != converted)

    # With 37 words and 20%, math.ceil(37 * 0.2) = 8 words should be converted
    expected_conversion_count = math.ceil(word_count * 0.2)
    assert converted_count == expected_conversion_count
