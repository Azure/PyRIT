# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import pytest

from pyrit.prompt_converter.charswap_attack_converter import CharSwapGenerator

from unittest.mock import patch


# Test that the converter produces the expected number of outputs
@pytest.mark.asyncio
async def test_char_swap_generator_output_count():
    converter = CharSwapGenerator(max_iterations=5)
    prompt = "This is a test prompt for the char swap generator."
    result = await converter.convert_async(prompt=prompt)
    output_prompts = result.output_text.strip().split("\n")
    assert len(output_prompts) == 5  # Should generate 5 perturbed prompts


# Test that words longer than 3 characters are being perturbed
@pytest.mark.asyncio
async def test_char_swap_generator_word_perturbation():
    converter = CharSwapGenerator(max_iterations=1, word_swap_ratio=1.0)
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
async def test_char_swap_generator_short_words(prompt):
    converter = CharSwapGenerator(max_iterations=1, word_swap_ratio=1.0)
    result = await converter.convert_async(prompt=prompt)
    output_prompts = result.output_text.strip().split("\n")
    # Since all words are <= 3 letters, output should be the same as input
    assert output_prompts[0] == prompt


# Test that punctuation is not perturbed
@pytest.mark.asyncio
async def test_char_swap_generator_punctuation():
    converter = CharSwapGenerator(max_iterations=1, word_swap_ratio=1.0)
    prompt = "Hello, world!"
    result = await converter.convert_async(prompt=prompt)
    output_prompts = result.output_text.strip().split("\n")
    # Punctuation should not be perturbed
    assert "," in output_prompts[0]
    assert "!" in output_prompts[0]


# Test that input type not supported raises ValueError
@pytest.mark.asyncio
async def test_char_swap_generator_input_type():
    converter = CharSwapGenerator()
    with pytest.raises(ValueError):
        await converter.convert_async(prompt="Test prompt", input_type="unsupported")


# Test with zero iterations
@pytest.mark.asyncio
async def test_char_swap_generator_zero_iterations():
    converter = CharSwapGenerator(max_iterations=0)
    prompt = "This is a test."
    result = await converter.convert_async(prompt=prompt)
    output_prompts = result.output_text.strip().split("\n")
    assert output_prompts == [""]  # Output should be an empty string in a list


# Test with word_swap_ratio=0
@pytest.mark.asyncio
async def test_char_swap_generator_zero_word_swap_ratio():
    converter = CharSwapGenerator(max_iterations=1, word_swap_ratio=0.0)
    prompt = "This is a test."
    result = await converter.convert_async(prompt=prompt)
    output_prompts = result.output_text.strip().split("\n")
    assert output_prompts[0] == prompt  # No words should be perturbed


@pytest.mark.asyncio
async def test_char_swap_generator_word_swap_ratio_other_than_1():
    converter = CharSwapGenerator(max_iterations=1, word_swap_ratio=0.5)
    prompt = "Testing word swap ratio"
    result = await converter.convert_async(prompt=prompt)
    output_prompts = result.output_text.strip().split("\n")
    assert output_prompts[0] != prompt


# Test that swapping is happening randomly
@pytest.mark.asyncio
async def test_char_swap_generator_random_swapping():
    converter = CharSwapGenerator(max_iterations=1, word_swap_ratio=1.0)
    prompt = "Character swapping test"
    result1 = await converter.convert_async(prompt=prompt)
    result2 = await converter.convert_async(prompt=prompt)
    # With randomness, result1 and result2 may be different
    assert result1.output_text != result2.output_text