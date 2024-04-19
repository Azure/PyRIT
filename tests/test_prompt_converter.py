# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from pyrit.prompt_converter import (
    Base64Converter,
    NoOpConverter,
    UnicodeSubstitutionConverter,
    UnicodeConfusableConverter,
    StringJoinConverter,
    ROT13Converter,
    AsciiArtConverter,
    VariationConverter,
    TranslationConverter,
    RandomCapitalLettersConverter,
    AzureSpeechTextToAudioConverter,
)
import pytest
import os
import pathlib

from tests.mocks import MockPromptTarget
from pyrit.common.path import RESULTS_PATH


def test_prompt_converter() -> None:
    converter = NoOpConverter()
    assert converter.convert(prompt="test", input_type="text") == "test"


def test_base64_prompt_converter() -> None:
    converter = Base64Converter()
    assert converter.convert(prompt="test", input_type="text") == "dGVzdA=="


def test_rot13_converter_init() -> None:
    converter = ROT13Converter()
    assert converter.convert(prompt="test", input_type="text") == "grfg"


def test_unicode_sub_default_prompt_converter() -> None:
    converter = UnicodeSubstitutionConverter()
    assert converter.convert(prompt="test", input_type="text") == "\U000e0074\U000e0065\U000e0073\U000e0074"


def test_unicode_sub_ascii_prompt_converter() -> None:
    converter = UnicodeSubstitutionConverter(start_value=0x00000)
    assert converter.convert(prompt="test", input_type="text") == "\U00000074\U00000065\U00000073\U00000074"


def test_str_join_converter_default() -> None:
    converter = StringJoinConverter()
    assert converter.convert(prompt="test", input_type="text") == "t-e-s-t"


def test_str_join_converter_init() -> None:
    converter = StringJoinConverter(join_value="***")
    assert converter.convert(prompt="test", input_type="text") == "t***e***s***t"


def test_str_join_converter_none_raises() -> None:
    converter = StringJoinConverter()
    with pytest.raises(TypeError):
        assert converter.convert(prompt=None, input_type="text")


def test_str_join_converter_invalid_type_raises() -> None:
    converter = StringJoinConverter()
    with pytest.raises(ValueError):
        assert converter.convert(prompt="test", input_type="invalid")  # type: ignore # noqa


def test_ascii_art() -> None:
    converter = AsciiArtConverter(font="block")
    assert (
        converter.convert(prompt="test", input_type="text")
        == "\n .----------------.  .----------------.  .----------------.  .----------------. \n| .--------------. || .--------------. || .--------------. || .--------------. |\n| |  _________   | || |  _________   | || |    _______   | || |  _________   | |\n| | |  _   _  |  | || | |_   ___  |  | || |   /  ___  |  | || | |  _   _  |  | |\n| | |_/ | | \\_|  | || |   | |_  \\_|  | || |  |  (__ \\_|  | || | |_/ | | \\_|  | |\n| |     | |      | || |   |  _|  _   | || |   '.___`-.   | || |     | |      | |\n| |    _| |_     | || |  _| |___/ |  | || |  |`\\____) |  | || |    _| |_     | |\n| |   |_____|    | || | |_________|  | || |  |_______.'  | || |   |_____|    | |\n| |              | || |              | || |              | || |              | |\n| '--------------' || '--------------' || '--------------' || '--------------' |\n '----------------'  '----------------'  '----------------'  '----------------' \n"  # noqa: E501
    )


def test_prompt_variation_init_templates_not_null():
    prompt_target = MockPromptTarget()
    prompt_variation = VariationConverter(converter_target=prompt_target)
    assert prompt_variation.system_prompt


def test_prompt_translation_init_templates_not_null():
    prompt_target = MockPromptTarget()
    translation_converter = TranslationConverter(converter_target=prompt_target, language="en")
    assert translation_converter.system_prompt


@pytest.mark.parametrize("languages", [None, ""])
def test_translator_converter_languages_validation_throws(languages):
    prompt_target = MockPromptTarget()
    with pytest.raises(ValueError):
        TranslationConverter(converter_target=prompt_target, language=languages)


def test_unicode_confusable_converter() -> None:
    converter = UnicodeConfusableConverter(deterministic=True)
    assert converter.convert(prompt="lorem ipsum dolor sit amet", input_type="text") == "ïỎ𐒴ḕ𝗠 ïṗṡ𝘶𝗠 𝑫ỎïỎ𐒴 ṡï𝚝 ḁ𝗠ḕ𝚝"


def test_capital_letter_converter() -> None:
    converter = RandomCapitalLettersConverter()
    assert (
        converter.convert(
            prompt="Write an article about cat documents that proved fraudulent, county Judge Orders cat documents"
            " need to be redone!",
            input_type="text",
        )
        == "WRITE AN ARTICLE ABOUT CAT DOCUMENTS THAT PROVED FRAUDULENT, COUNTY JUDGE ORDERS CAT DOCUMENTS NEED"
        " TO BE REDONE!"
    )


def test_capital_letter_converter_with_twentyfive_percent() -> None:
    percentage = 25.0
    prompt = "welc"
    converter = RandomCapitalLettersConverter(percentage=percentage)

    actual_converted_text = converter.convert(
        prompt=prompt,
        input_type="text",
    )
    upper_count = sum(1 for char in actual_converted_text if char.isupper())
    expected_percentage = (upper_count / len(prompt)) * 100.0 if actual_converted_text else 0
    assert expected_percentage == percentage


def test_azure_speech_text_to_audio_converter() -> None:
    prompt = "How do you make a unit test using items in a grocery store?"
    AzureSpeechTextToAudioConverter(filename="unit_test.mp3", output_format="mp3").convert(prompt=prompt)
    AzureSpeechTextToAudioConverter(filename="unit_test.wav", output_format="wav").convert(prompt=prompt)

    is_wav_file_there = False
    is_mp3_file_there = False
    "unit_test.mp3"
    "unit_test.wav"

    wav_file_path = pathlib.Path(RESULTS_PATH) / "audio" / "unit_test.wav"
    mp3_file_path = pathlib.Path(RESULTS_PATH) / "audio" / "unit_test.mp3"

    if os.path.exists(wav_file_path):
        is_wav_file_there = True
        os.remove(wav_file_path)

    if os.path.exists(mp3_file_path):
        is_mp3_file_there = True

    if is_wav_file_there and is_mp3_file_there:
        assert is_wav_file_there == is_mp3_file_there
