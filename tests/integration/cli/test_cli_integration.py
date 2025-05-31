# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import re
import shlex
import tempfile
from typing import Collection, Literal

import pytest
import yaml

from pyrit.cli.__main__ import main

test_cases_success = [
    "--config-file 'tests/integration/cli/mixed_multiple_orchestrators_args_success.yaml'",
    "--config-file 'tests/integration/cli/prompt_send_success.yaml'",
    "--config-file 'tests/unit/cli/prompt_send_success_converters_custom_target.yaml'",
    "--config-file 'tests/unit/cli/prompt_send_success_converters_no_target.yaml'",
    "--config-file 'tests/integration/cli/multiple_converters_success.yaml'",
]


@pytest.mark.parametrize("command", test_cases_success)
def test_cli_integration_success(command):
    main(shlex.split(command))


converters = [
    (
        "text",
        {"type": "AnsiAttackConverter"},
    ),
    (
        "text",
        {"type": "AddImageTextConverter", "img_to_add": "./assets/pyrit_architecture.png"},
    ),
    (
        "text",
        {"type": "AsciiArtConverter"},
    ),
    (
        "text",
        {"type": "AsciiSmugglerConverter"},
    ),
    (
        "text",
        {"type": "AtbashConverter"},
    ),
    (
        "text",
        {"type": "AzureSpeechTextToAudioConverter"},
    ),
    (
        "text",
        {"type": "Base64Converter"},
    ),
    (
        "text",
        {"type": "BinaryConverter"},
    ),
    (
        "text",
        {"type": "CaesarConverter", "caesar_offset": 3},
    ),
    (
        "text",
        {"type": "CharacterSpaceConverter"},
    ),
    (
        "text",
        {"type": "CharSwapGenerator"},
    ),
    (
        "text",
        {"type": "CodeChameleonConverter", "encrypt_type": "reverse"},
    ),
    (
        "text",
        {"type": "ColloquialWordswapConverter"},
    ),
    ("text", {"type": "DenylistConverter", "denylist": ["Molotov"]}),
    (
        "text",
        {"type": "DiacriticConverter"},
    ),
    (
        "text",
        {"type": "EmojiConverter"},
    ),
    (
        "text",
        {"type": "FlipConverter"},
    ),
    (
        "text",
        {"type": "InsertPunctuationConverter"},
    ),
    (
        "text",
        {"type": "LeetspeakConverter"},
    ),
    (
        "text",
        {"type": "MaliciousQuestionGeneratorConverter"},
    ),
    (
        "text",
        {"type": "MathPromptConverter"},
    ),
    (
        "text",
        {"type": "MorseConverter"},
    ),
    (
        "text",
        {"type": "NoiseConverter"},
    ),
    (
        "text",
        {"type": "PDFConverter"},
    ),
    (
        "text",
        {"type": "PersuasionConverter", "persuasion_technique": "misrepresentation"},
    ),
    (
        "text",
        {"type": "QRCodeConverter"},
    ),
    (
        "text",
        {"type": "RandomCapitalLettersConverter"},
    ),
    (
        "text",
        {"type": "RepeatTokenConverter", "token_to_repeat": "test", "times_to_repeat": 2},
    ),
    (
        "text",
        {"type": "ROT13Converter"},
    ),
    (
        "text",
        {"type": "SearchReplaceConverter", "pattern": "test", "replace": "TEST"},
    ),
    (
        "text",
        {"type": "StringJoinConverter"},
    ),
    (
        "text",
        {"type": "SuffixAppendConverter", "suffix": "!!!"},
    ),
    (
        "text",
        {"type": "TenseConverter", "tense": "past"},
    ),
    (
        "text",
        {"type": "TextToHexConverter"},
    ),
    (
        "text",
        {"type": "ToneConverter", "tone": "sarcastic"},
    ),
    (
        "text",
        {"type": "ToxicSentenceGeneratorConverter"},
    ),
    (
        "text",
        {"type": "TranslationConverter", "language": "spanish"},
    ),
    (
        "text",
        {"type": "UnicodeConfusableConverter"},
    ),
    (
        "text",
        {"type": "UnicodeReplacementConverter"},
    ),
    (
        "text",
        {"type": "UnicodeSubstitutionConverter"},
    ),
    (
        "text",
        {"type": "UrlConverter"},
    ),
    (
        "text",
        {"type": "VariationConverter"},
    ),
    (
        "text",
        {"type": "ZalgoConverter"},
    ),
    (
        "text",
        {"type": "ZeroWidthConverter"},
    ),
    (
        "image",
        {"type": "AddTextImageConverter", "text_to_add": "test"},
    ),
    (
        "image",
        {"type": "AddImageVideoConverter", "video_path": "./assets/sample_video.mp4"},
    ),
    (
        "audio",
        {"type": "AudioFrequencyConverter"},
    ),
    (
        "audio",
        {"type": "AzureSpeechAudioToTextConverter"},
    ),
]


def _create_data(data_type: str) -> dict[str, Collection[Collection[str]]]:
    data = {
        "scenarios": [{"type": "PromptSendingOrchestrator"}],
        "adversarial_chat": {"type": "OpenAIChatTarget"},
        "objective_target": {"type": "OpenAIChatTarget"},
        "database": {"type": "DuckDB"},
    }

    if data_type == "text":
        data["datasets"] = ["./tests/integration/cli/illegal_integration_test.prompt"]
    elif data_type == "image":
        data["datasets"] = ["./tests/integration/cli/illegal_integration_test_image_input.prompt"]
    elif data_type == "audio":
        data["datasets"] = ["./tests/integration/cli/illegal_integration_test_audio_input.prompt"]

    return data


@pytest.mark.parametrize("input_type, converter_data", converters)
def test_cli_integration_success_converters_all(
    input_type: Literal["text", "image", "audio"], converter_data: dict[str, str]
):
    # scenario: verify converter functionality with all converters as of v0.9.0
    # adversarial_chat available as default converter_target when necessary
    # converters that do not need a target also included
    data = _create_data(input_type)
    data["converters"] = [converter_data]

    use_text_target_types = ["PDFConverter", "AddImageVideoConverter", "AudioFrequencyConverter"]
    if converter_data["type"] in use_text_target_types:
        # OpenAIChatTarget only allows text and image_path data types as input, and these converters
        # output different data types so we need to use TextTarget
        data["objective_target"] = {"type": "TextTarget"}

    # create temp yaml file for each converter
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".yaml") as temp_file:
        yaml.dump(data, temp_file)
        main(shlex.split(s=f"--config-file '{temp_file.name}'"))

        # clean up temp file
        temp_file.close()
        os.remove(temp_file.name)


test_cases_error = [
    (
        "--config-file 'tests/integration/cli/prompt_send_converters_wrong_data_type.yaml'",
        "Input type not supported",
        ValueError,
    ),
]


@pytest.mark.parametrize("command, expected_output, error_type", test_cases_error)
def test_cli_integration_error(command, expected_output, error_type):
    with pytest.raises(error_type, match=re.escape(expected_output)):
        main(shlex.split(command))
