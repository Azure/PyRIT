# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from unittest.mock import MagicMock

from pyrit.prompt_converter import (
    NoiseConverter,
    TenseConverter,
    ToneConverter,
)


def test_noise_converter_sets_system_prompt_default() -> None:
    target = MagicMock()
    converter = NoiseConverter(converter_target=target)
    assert "Grammar error, Delete random letter" in converter.system_prompt


def test_noise_converter_sets_system_prompt() -> None:
    target = MagicMock()
    converter = NoiseConverter(converter_target=target, noise="extra random periods")
    assert "extra random periods" in converter.system_prompt


def test_tone_converter_sets_system_prompt() -> None:
    target = MagicMock()
    converter = ToneConverter(tone="formal", converter_target=target)
    assert "formal" in converter.system_prompt


def test_tense_converter_sets_system_prompt() -> None:
    target = MagicMock()
    converter = TenseConverter(tense="past", converter_target=target)
    assert "past" in converter.system_prompt
