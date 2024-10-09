# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

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
    word = "xxxx"  # A word with no expected homoglyphs
    variants = homoglyph_generator._get_homoglyph_variants(word)
    assert len(variants) == 0 or "᙮" in variants


def test_generate_perturbed_prompts(homoglyph_generator):
    prompt = "This is a test."
    perturbed_prompt = homoglyph_generator._generate_perturbed_prompts(prompt)
    assert isinstance(perturbed_prompt, str)
    # Ensure that the perturbed prompt is different from the original
    assert perturbed_prompt != prompt


@pytest.mark.asyncio
async def test_convert_async(homoglyph_generator):
    prompt = "This is a test."
    result = await homoglyph_generator.convert_async(prompt=prompt, input_type="text")
    assert isinstance(result, ConverterResult)
    assert result.output_type == "text"
    perturbed_prompts = result.output_text.split("\n")
    assert len(perturbed_prompts) <= homoglyph_generator.max_iterations
    for p_prompt in perturbed_prompts:
        assert p_prompt != prompt
        assert isinstance(p_prompt, str)


@pytest.mark.asyncio
async def test_input_not_supported(homoglyph_generator):
    with pytest.raises(ValueError):
        await homoglyph_generator.convert_async(prompt="This is a test.", input_type="image")


@pytest.mark.asyncio
async def test_convert_async_non_ascii_word(homoglyph_generator):
    prompt = "café"
    result = await homoglyph_generator.convert_async(prompt=prompt, input_type="text")

    assert isinstance(result, ConverterResult)
    perturbed_prompt = result.output_text

    # Assert that we are getting perturbed results of the correct type and that it's not the same as original
    assert isinstance(perturbed_prompt, str)
    assert perturbed_prompt != prompt, "The perturbed prompt should be different from the original prompt."

    # Check if non-ASCII character 'é' is handled correctly
    assert "é" in perturbed_prompt or "e" in perturbed_prompt, "The non-ASCII character 'é' should be handled properly."


@patch(
    "pyrit.prompt_converter.homoglyph_generator_converter.is_confusable",
    side_effect=UnicodeDecodeError("codec", b"", 0, 1, "reason"),
)
def test_get_homoglyph_variants_unicode_error(mock_is_confusable, homoglyph_generator):
    # Test to check homoglyphs when UnicodeDecodeError is raised
    variants = homoglyph_generator._get_homoglyph_variants("test")
    assert variants == []  # Ensure the result is an empty list when the exception is raised


@pytest.mark.asyncio
async def test_convert_async_empty_prompt(homoglyph_generator):
    prompt = ""
    result = await homoglyph_generator.convert_async(prompt=prompt, input_type="text")
    assert isinstance(result, ConverterResult)
    assert result.output_text == ""


@pytest.mark.asyncio
async def test_convert_async_long_prompt(homoglyph_generator):
    prompt = (
        "This is a long prompt intended to test how the HomoglyphGenerator handles larger inputs. "
        "It contains multiple sentences, various punctuation marks, and a mix of short and long words."
    )
    result = await homoglyph_generator.convert_async(prompt=prompt, input_type="text")
    assert isinstance(result, ConverterResult)
    perturbed_prompts = result.output_text.split("\n")
    assert len(perturbed_prompts) <= homoglyph_generator.max_iterations
    for p_prompt in perturbed_prompts:
        assert p_prompt != prompt
        assert isinstance(p_prompt, str)
