# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import re

import pytest

from pyrit.prompt_converter.colloquial_wordswap_converter import (
    ColloquialWordswapConverter,
)


# Test for deterministic mode
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "input_text,expected_output",
    [
        ("grandfather", "ah gong"),  # Single wordswap
        ("mother and brother", "mama and bro"),  # Default substitution for mother and brother
        ("Hello, my Father!", "Hello, my papa!"),  # Combined substitutions with punctuation
    ],
)
async def test_colloquial_deterministic(input_text, expected_output):
    converter = ColloquialWordswapConverter(deterministic=True)
    result = await converter.convert_async(prompt=input_text)
    assert result.output_text == expected_output


# Test for non-deterministic mode
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "input_text",
    [
        "code",  # A different input set to reduce redundancy
        "mother",
        "uncle and brother",
    ],
)
async def test_colloquial_non_deterministic(input_text):
    converter = ColloquialWordswapConverter(deterministic=False)
    result = await converter.convert_async(prompt=input_text)

    # Valid substitution mappings in the input texts
    valid_substitutions = {
        "mother": ["mama", "amma", "ibu"],
        "uncle": ["encik", "unker"],
        "brother": ["bro", "boiboi", "di di", "xdd", "anneh", "thambi"],
    }

    # Split input and output into words, preserving multi-word substitutions as single tokens
    input_words = re.findall(r"\w+|\S+", input_text)
    output_words = re.findall(r"\w+|\S+", result.output_text)

    # Check that each wordswap is a valid substitution
    for input_word, output_word in zip(input_words, output_words):
        lower_input_word = input_word.lower()

        if lower_input_word in valid_substitutions:
            assert any(sub in output_word or output_word in sub for sub in valid_substitutions[lower_input_word])
        else:
            assert output_word == input_word


# Test for custom substitutions
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "input_text,custom_substitutions,expected_output",
    [
        ("father", {"father": ["appa", "darth vader"]}, "appa"),  # Custom substitution father -> appa
    ],
)
async def test_colloquial_custom_substitutions(input_text, custom_substitutions, expected_output):
    converter = ColloquialWordswapConverter(deterministic=True, custom_substitutions=custom_substitutions)
    result = await converter.convert_async(prompt=input_text)
    assert result.output_text == expected_output


# Test for empty custom substitutions
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "input_text,expected_output",
    [
        ("mother and father", "mama and papa"),  # Using default substitutions when custom is empty
    ],
)
async def test_colloquial_empty_custom_substitutions(input_text, expected_output):
    converter = ColloquialWordswapConverter(deterministic=True, custom_substitutions={})
    result = await converter.convert_async(prompt=input_text)
    assert result.output_text == expected_output


# Test multiple word prompts
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "input_text,expected_output",
    [
        ("father and mother", "papa and mama"),
        ("brother and sister", "bro and xjj"),
        ("aunt and uncle", "makcik and encik"),
    ],
)
async def test_multiple_words(input_text, expected_output):
    converter = ColloquialWordswapConverter(deterministic=True)
    result = await converter.convert_async(prompt=input_text)
    assert result.output_text == expected_output


# Test for awkward spacing
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "input_text,expected_output",
    [
        ("  father  and    mother ", "papa and mama"),
        ("sister   and   brother", "xjj and bro"),
    ],
)
async def test_awkward_spacing(input_text, expected_output):
    converter = ColloquialWordswapConverter(deterministic=True)
    result = await converter.convert_async(prompt=input_text)
    assert result.output_text == expected_output


# Test for punctuation handling
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "input_text,expected_output",
    [
        ("father, mother!", "papa, mama!"),
        ("aunt? uncle!", "makcik? encik!"),
    ],
)
async def test_punctuation_handling(input_text, expected_output):
    converter = ColloquialWordswapConverter(deterministic=True)
    result = await converter.convert_async(prompt=input_text)
    assert result.output_text == expected_output


def test_colloquial_converter_input_supported() -> None:
    converter = ColloquialWordswapConverter()
    assert converter.input_supported("text") is True
    assert converter.input_supported("image_path") is False
