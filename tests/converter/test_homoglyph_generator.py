# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import asyncio
from unittest.mock import patch

import pytest

from pyrit.prompt_converter import ConverterResult
from pyrit.prompt_converter.homoglyph_generator_converter import HomoglyphGenerator


@pytest.fixture
def homoglyph_generator():
    return HomoglyphGenerator(max_iterations=5)


def test_input_supported(homoglyph_generator):
    assert homoglyph_generator.input_supported("text") is True
    assert homoglyph_generator.input_supported("image") is False


def test_get_homoglyph_variants(homoglyph_generator):
    word = "test"
    variants = homoglyph_generator._get_homoglyph_variants(word)
    assert isinstance(variants, list)
    assert word not in variants  # Original word should not be in variants
    # Since homoglyph variants depend on the external library and mappings,
    # we cannot assert exact variants, but we can check that variants are different
    for variant in variants:
        assert variant != word


def test_get_homoglyph_variants_no_variants(homoglyph_generator):
    word = "88888888"  # Assuming no homoglyphs for the digit '8'
    variants = homoglyph_generator._get_homoglyph_variants(word)
    assert variants == []


def test_generate_perturbed_prompts(homoglyph_generator):
    prompt = "This is a test."
    perturbed_prompts = homoglyph_generator._generate_perturbed_prompts(prompt)
    assert isinstance(perturbed_prompts, list)
    assert len(perturbed_prompts) <= homoglyph_generator.max_iterations
    # Each perturbed prompt should be different from the original
    for p_prompt in perturbed_prompts:
        assert p_prompt != prompt
        assert isinstance(p_prompt, str)
    # The number of perturbed prompts should be as expected
    # Since this depends on the actual homoglyph mappings, we can't assert exact numbers


@pytest.mark.asyncio
async def test_convert_async(homoglyph_generator):
    prompt = "This is a test."
    result = await homoglyph_generator.convert_async(prompt=prompt, input_type="text")
    assert isinstance(result, ConverterResult)
    assert result.output_type == "text"
    perturbed_prompts = result.output_text.split('\n')
    assert len(perturbed_prompts) <= homoglyph_generator.max_iterations
    for p_prompt in perturbed_prompts:
        assert p_prompt != prompt
        assert isinstance(p_prompt, str)


def test_input_not_supported(homoglyph_generator):
    with pytest.raises(ValueError):
        asyncio.run(homoglyph_generator.convert_async(prompt="This is a test.", input_type="image"))


@pytest.mark.asyncio
async def test_convert_async_non_ascii_word(homoglyph_generator):
    prompt = "This is a test with non-ASCII character: café."
    result = await homoglyph_generator.convert_async(prompt=prompt, input_type="text")
    assert isinstance(result, ConverterResult)
    perturbed_prompts = result.output_text.split('\n')
    for p_prompt in perturbed_prompts:
        assert isinstance(p_prompt, str)
        # Ensure the non-ASCII word is handled appropriately
        assert 'café' in p_prompt or 'cafe' in p_prompt


def test_get_homoglyph_variants_unicode_error(homoglyph_generator):
    # Mock the homoglyphs.to_ascii method to raise UnicodeDecodeError
    with patch.object(
        homoglyph_generator.homoglyphs,
        'to_ascii',
        side_effect=UnicodeDecodeError('codec', b'', 0, 1, 'reason')
    ):
        variants = homoglyph_generator._get_homoglyph_variants('test')
        assert variants == []


@pytest.mark.asyncio
async def test_convert_async_empty_prompt(homoglyph_generator):
    prompt = ""
    result = await homoglyph_generator.convert_async(prompt=prompt, input_type="text")
    assert isinstance(result, ConverterResult)
    assert result.output_text == ""


@ pytest.mark.asyncio
async def test_convert_async_long_prompt(homoglyph_generator):
    prompt = "This is a long prompt intended to test how the HomoglyphGenerator handles larger inputs. " \
             "It contains multiple sentences, various punctuation marks, and a mix of short and long words."
    result = await homoglyph_generator.convert_async(prompt=prompt, input_type="text")
    assert isinstance(result, ConverterResult)
    perturbed_prompts = result.output_text.split('\n')
    assert len(perturbed_prompts) <= homoglyph_generator.max_iterations
    for p_prompt in perturbed_prompts:
        assert p_prompt != prompt
        assert isinstance(p_prompt, str)
