# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from pyrit.prompt_converter import (
    Base64Converter,
    UnicodeSubstitutionConverter,
    UnicodeConfusableConverter,
    StringJoinConverter,
    ROT13Converter,
    AsciiArtConverter,
    VariationConverter,
    TranslationConverter,
    RandomCapitalLettersConverter,
    SearchReplaceConverter,
    LeetspeakConverter,
)
import pytest

from tests.mocks import MockPromptTarget


def test_base64_prompt_converter() -> None:
    converter = Base64Converter()
    output = converter.convert(prompt="test", input_type="text")
    assert output.output_text == "dGVzdA=="
    assert output.output_type == "text"


def test_rot13_converter_init() -> None:
    converter = ROT13Converter()
    output = converter.convert(prompt="test", input_type="text")
    assert output.output_text == "grfg"
    assert output.output_type == "text"


def test_unicode_sub_default_prompt_converter() -> None:
    converter = UnicodeSubstitutionConverter()
    output = converter.convert(prompt="test", input_type="text")
    assert output.output_text == "\U000e0074\U000e0065\U000e0073\U000e0074"
    assert output.output_type == "text"


def test_unicode_sub_ascii_prompt_converter() -> None:
    converter = UnicodeSubstitutionConverter(start_value=0x00000)
    output = converter.convert(prompt="test", input_type="text")
    assert output.output_text == "\U00000074\U00000065\U00000073\U00000074"
    assert output.output_type == "text"


def test_str_join_converter_default() -> None:
    converter = StringJoinConverter()
    output = converter.convert(prompt="test", input_type="text")
    assert output.output_text == "t-e-s-t"
    assert output.output_type == "text"


def test_str_join_converter_init() -> None:
    converter = StringJoinConverter(join_value="***")
    output = converter.convert(prompt="test", input_type="text")
    assert output.output_text == "t***e***s***t"
    assert output.output_type == "text"


def test_str_join_converter_none_raises() -> None:
    converter = StringJoinConverter()
    with pytest.raises(TypeError):
        assert converter.convert(prompt=None, input_type="text")


def test_str_join_converter_invalid_type_raises() -> None:
    converter = StringJoinConverter()
    with pytest.raises(ValueError):
        assert converter.convert(prompt="test", input_type="invalid")  # type: ignore # noqa


def test_str_join_converter_unsupported_type_raises() -> None:
    converter = StringJoinConverter()
    with pytest.raises(ValueError):
        assert converter.convert(prompt="test", input_type="image_path")


def test_ascii_art() -> None:
    converter = AsciiArtConverter(font="block")
    output = converter.convert(prompt="test", input_type="text")

    assert output.output_text == (
        "\n .----------------.  .----------------.  .----------------.  .----------------. \n| .--------------. || .--------------. || .--------------. || .--------------. |\n| |  _________   | || |  _________   | || |    _______   | || |  _________   | |\n| | |  _   _  |  | || | |_   ___  |  | || |   /  ___  |  | || | |  _   _  |  | |\n| | |_/ | | \\_|  | || |   | |_  \\_|  | || |  |  (__ \\_|  | || | |_/ | | \\_|  | |\n| |     | |      | || |   |  _|  _   | || |   '.___`-.   | || |     | |      | |\n| |    _| |_     | || |  _| |___/ |  | || |  |`\\____) |  | || |    _| |_     | |\n| |   |_____|    | || | |_________|  | || |  |_______.'  | || |   |_____|    | |\n| |              | || |              | || |              | || |              | |\n| '--------------' || '--------------' || '--------------' || '--------------' |\n '----------------'  '----------------'  '----------------'  '----------------' \n"  # noqa: E501
    )
    assert output.output_type == "text"


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
    output = converter.convert(prompt="lorem ipsum dolor sit amet", input_type="text")
    assert output.output_text == "ïỎ𐒴ḕ𝗠 ïṗṡ𝘶𝗠 𝑫ỎïỎ𐒴 ṡï𝚝 ḁ𝗠ḕ𝚝"
    assert output.output_type == "text"


def test_character_replacement_converter() -> None:
    converter = SearchReplaceConverter(old_value=" ", new_value="_")
    output = converter.convert(prompt="Hello World !", input_type="text")
    assert output.output_text == "Hello_World_!"
    assert output.output_type == "text"


def test_leetcode_converter() -> None:
    converter = LeetspeakConverter()
    output = converter.convert(prompt="mood", input_type="text")
    assert output.output_text == "m00d"
    assert output.output_type == "text"


def test_capital_letter_converter() -> None:
    converter = RandomCapitalLettersConverter()
    output = converter.convert(
        prompt="Write an article about cat documents that proved fraudulent, county Judge Orders cat documents"
        " need to be redone!",
        input_type="text",
    )

    assert (
        output.output_text
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
    ).output_text

    upper_count = sum(1 for char in actual_converted_text if char.isupper())
    expected_percentage = (upper_count / len(prompt)) * 100.0 if actual_converted_text else 0
    assert expected_percentage == percentage
