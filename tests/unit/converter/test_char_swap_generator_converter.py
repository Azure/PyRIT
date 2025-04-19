# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from unittest.mock import patch

import pytest

from pyrit.prompt_converter.charswap_attack_converter import CharSwapConverter


# Test that the converter produces the expected number of outputs
@pytest.mark.asyncio
async def test_char_swap_converter_output_count():
    converter = CharSwapConverter(max_iterations=5)
    prompt = "This is a test prompt for the char swap converter."
    result = await converter.convert_async(prompt=prompt)
    output_prompts = result.output_text.strip().split("\n")
    assert len(output_prompts) == 1  # Should generate 1 perturbed prompt


# Test that words longer than 3 characters are being perturbed
@pytest.mark.asyncio
async def test_char_swap_converter_word_perturbation():
    converter = CharSwapConverter(max_iterations=1, mode="random", percentage=100)
    prompt = "Testing"
    with patch("random.randint", return_value=1):  # Force swap at position 1
        result = await converter.convert_async(prompt=prompt)
        output_prompts = result.output_text.strip().split("\n")
        assert output_prompts[0] == "Tseting"  # 'Testing' with 'e' and 's' swapped


# Test that words of length <= 3 are not perturbed
@pytest.mark.parametrize(
    "prompt",
    ["Try or do?", "To be or not to be.", "2b oR n0t 2b"],
)
@pytest.mark.asyncio
async def test_char_swap_converter_short_words(prompt):
    converter = CharSwapConverter(max_iterations=1, mode="random", percentage=100)
    result = await converter.convert_async(prompt=prompt)
    output_prompts = result.output_text.strip().split("\n")
    # Since all words are <= 3 letters, output should be the same as input
    assert output_prompts[0] == prompt


# Test that punctuation is not perturbed
@pytest.mark.asyncio
async def test_char_swap_converter_punctuation():
    converter = CharSwapConverter(max_iterations=1, mode="random", percentage=100)
    prompt = "Hello, world!"
    result = await converter.convert_async(prompt=prompt)
    output_prompts = result.output_text.strip().split("\n")
    # Punctuation should not be perturbed
    assert "," in output_prompts[0]
    assert "!" in output_prompts[0]


# Test that input type not supported raises ValueError
@pytest.mark.asyncio
async def test_char_swap_converter_input_type():
    converter = CharSwapConverter()
    with pytest.raises(ValueError):
        await converter.convert_async(prompt="Test prompt", input_type="unsupported")


# Test with zero iterations
@pytest.mark.asyncio
async def test_char_swap_converter_zero_iterations():
    with pytest.raises(ValueError, match="max_iterations must be greater than 0"):
        CharSwapConverter(max_iterations=0)


@pytest.mark.asyncio
async def test_char_swap_converter_sample_ratio_other_than_1():
    converter = CharSwapConverter(max_iterations=1, mode="random", percentage=50)
    prompt = "Testing word swap ratio"
    result = await converter.convert_async(prompt=prompt)
    output_prompts = result.output_text.strip().split("\n")
    assert output_prompts[0] != prompt


# Test that swapping is happening randomly
@pytest.mark.asyncio
async def test_char_swap_converter_random_swapping():
    converter = CharSwapConverter(max_iterations=1, mode="random", percentage=100)
    prompt = "Character swapping test"

    with patch(
        "random.sample",
        side_effect=[[0, 1, 2]],
    ):
        result1 = await converter.convert_async(prompt=prompt)

    assert prompt != result1.output_text
