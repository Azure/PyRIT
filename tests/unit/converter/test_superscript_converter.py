# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import pytest

from pyrit.prompt_converter import ConverterResult, SuperscriptConverter


@pytest.mark.asyncio
async def test_superscript_converter():
    converter = SuperscriptConverter()
    prompts = ["Let's test this converter!", "Unsupported characters stay the same: qCFQSXYZ"]
    expected_outputs = ["ᴸᵉᵗ'ˢ ᵗᵉˢᵗ ᵗʰⁱˢ ᶜᵒⁿᵛᵉʳᵗᵉʳ!", "ᵁⁿˢᵘᵖᵖᵒʳᵗᵉᵈ ᶜʰᵃʳᵃᶜᵗᵉʳˢ ˢᵗᵃʸ ᵗʰᵉ ˢᵃᵐᵉ: qCFQSXYZ"]
    for prompt, expected_output in zip(prompts, expected_outputs):
        result = await converter.convert_async(prompt=prompt, input_type="text")
        assert isinstance(result, ConverterResult)
        assert result.output_text == expected_output
