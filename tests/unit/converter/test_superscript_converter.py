# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

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
