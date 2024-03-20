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
)
import pytest

from tests.mocks import MockPromptTarget


def test_prompt_converter() -> None:
    converter = NoOpConverter()
    assert converter.convert(["test"]) == ["test"]


def test_base64_prompt_converter() -> None:
    converter = Base64Converter()
    assert converter.convert(["test"]) == ["dGVzdA=="]


def test_rot13_converter_init() -> None:
    converter = ROT13Converter()
    assert converter.convert(["test"]) == ["grfg"]


def test_unicode_sub_default_prompt_converter() -> None:
    converter = UnicodeSubstitutionConverter()
    assert converter.convert(["test"]) == ["\U000e0074\U000e0065\U000e0073\U000e0074"]


def test_unicode_sub_ascii_prompt_converter() -> None:
    converter = UnicodeSubstitutionConverter(start_value=0x00000)
    assert converter.convert(["test"]) == ["\U00000074\U00000065\U00000073\U00000074"]


def test_str_join_converter_default() -> None:
    converter = StringJoinConverter()
    assert converter.convert(["test"]) == ["t-e-s-t"]


def test_str_join_converter_init() -> None:
    converter = StringJoinConverter(join_value="***")
    assert converter.convert(["test"]) == ["t***e***s***t"]


def test_str_join_converter_multi_word() -> None:
    converter = StringJoinConverter()
    assert converter.convert(["test1", "test2", "test3"]) == ["t-e-s-t-1", "t-e-s-t-2", "t-e-s-t-3"]


def test_str_join_converter_none_raises() -> None:
    converter = StringJoinConverter()
    with pytest.raises(TypeError):
        assert converter.convert(None)


def test_ascii_art() -> None:
    converter = AsciiArtConverter(font="block")
    assert converter.convert(["test"]) == [
        "\n .----------------.  .----------------.  .----------------.  .----------------. \n| .--------------. || .--------------. || .--------------. || .--------------. |\n| |  _________   | || |  _________   | || |    _______   | || |  _________   | |\n| | |  _   _  |  | || | |_   ___  |  | || |   /  ___  |  | || | |  _   _  |  | |\n| | |_/ | | \\_|  | || |   | |_  \\_|  | || |  |  (__ \\_|  | || | |_/ | | \\_|  | |\n| |     | |      | || |   |  _|  _   | || |   '.___`-.   | || |     | |      | |\n| |    _| |_     | || |  _| |___/ |  | || |  |`\\____) |  | || |    _| |_     | |\n| |   |_____|    | || | |_________|  | || |  |_______.'  | || |   |_____|    | |\n| |              | || |              | || |              | || |              | |\n| '--------------' || '--------------' || '--------------' || '--------------' |\n '----------------'  '----------------'  '----------------'  '----------------' \n"  # noqa: E501
    ]


def test_prompt_variation_init_templates_not_null():
    prompt_target = MockPromptTarget()
    prompt_variation = VariationConverter(converter_target=prompt_target)
    assert prompt_variation.system_prompt


def test_prompt_variation_init_templates_system():
    prompt_target = MockPromptTarget()
    prompt_variation = VariationConverter(converter_target=prompt_target, number_variations=20)
    assert "20 different responses" in prompt_variation.system_prompt


@pytest.mark.parametrize("number_variations,is_one_to_one", [(1, True), (2, False), (5, False)])
def test_prompt_variation_is_one_to_one(number_variations, is_one_to_one):
    prompt_target = MockPromptTarget()
    prompt_variation = VariationConverter(converter_target=prompt_target, number_variations=number_variations)
    assert prompt_variation.is_one_to_one_converter() == is_one_to_one


def test_prompt_translation_init_templates_not_null():
    prompt_target = MockPromptTarget()
    translation_converter = TranslationConverter(converter_target=prompt_target, languages=["en", "es"])
    assert translation_converter.system_prompt


@pytest.mark.parametrize("languages,is_one_to_one", [(["en", "es"], False), (["en"], True)])
def test_translator_converter_is_one_to_one(languages, is_one_to_one):
    prompt_target = MockPromptTarget()
    translation_converter = TranslationConverter(converter_target=prompt_target, languages=languages)
    assert translation_converter.is_one_to_one_converter() == is_one_to_one


@pytest.mark.parametrize("languages", [None, [], ["this, is my language"]])
def test_translator_converter_languages_validation_throws(languages):
    prompt_target = MockPromptTarget()
    with pytest.raises(ValueError):
        TranslationConverter(converter_target=prompt_target, languages=languages)


def test_unicode_confusable_converter() -> None:
    converter = UnicodeConfusableConverter(deterministic=True)
    assert converter.convert(["lorem ipsum dolor sit amet"]) == ["ïỎ𐒴ḕ𝗠 ïṗṡ𝘶𝗠 𝑫ỎïỎ𐒴 ṡï𝚝 ḁ𝗠ḕ𝚝"]
