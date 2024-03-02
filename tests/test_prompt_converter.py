# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from pyrit.prompt_converter import (
    Base64Converter,
    NoOpConverter,
    UnicodeSubstitutionConverter,
    StringJoinConverter,
    AsciiArtConverter,
)


def test_prompt_converter() -> None:
    converter = NoOpConverter()
    assert converter.convert("test") == "test"


def test_base64_prompt_converter() -> None:
    converter = Base64Converter()
    assert converter.convert("test") == "dGVzdA=="


def test_unicode_sub_default_prompt_converter() -> None:
    converter = UnicodeSubstitutionConverter()
    assert converter.convert("test") == "\U000e0074\U000e0065\U000e0073\U000e0074"


def test_unicode_sub_ascii_prompt_converter() -> None:
    converter = UnicodeSubstitutionConverter(0x00000)
    assert converter.convert("test") == "\U00000074\U00000065\U00000073\U00000074"


def test_str_join_converter_default() -> None:
    converter = StringJoinConverter()
    assert converter.convert("test") == "t-e-s-t"


def test_str_join_converter_init() -> None:
    converter = StringJoinConverter("***")
    assert converter.convert("test") == "t***e***s***t"


def test_ascii_art() -> None:
    converter = AsciiArtConverter("block")
    assert (
        converter.convert("test")
        == "\n .----------------.  .----------------.  .----------------.  .----------------. \n| .--------------. || .--------------. || .--------------. || .--------------. |\n| |  _________   | || |  _________   | || |    _______   | || |  _________   | |\n| | |  _   _  |  | || | |_   ___  |  | || |   /  ___  |  | || | |  _   _  |  | |\n| | |_/ | | \\_|  | || |   | |_  \\_|  | || |  |  (__ \\_|  | || | |_/ | | \\_|  | |\n| |     | |      | || |   |  _|  _   | || |   '.___`-.   | || |     | |      | |\n| |    _| |_     | || |  _| |___/ |  | || |  |`\\____) |  | || |    _| |_     | |\n| |   |_____|    | || | |_________|  | || |  |_______.'  | || |   |_____|    | |\n| |              | || |              | || |              | || |              | |\n| '--------------' || '--------------' || '--------------' || '--------------' |\n '----------------'  '----------------'  '----------------'  '----------------' \n"
    )
