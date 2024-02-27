# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from pyrit.prompt_converter import Base64Converter, NoOpConverter, UnicodeSubstitutionConverter


def test_prompt_transformer() -> None:
    transformer = NoOpConverter()
    assert transformer.convert("test") == "test"


def test_base64_prompt_transformer() -> None:
    transformer = Base64Converter()
    assert transformer.convert("test") == "dGVzdA=="


def test_unicode_sub_default_prompt_transformer() -> None:
    transformer = UnicodeSubstitutionConverter()
    assert transformer.convert("test") == "\U000e0074\U000e0065\U000e0073\U000e0074"


def test_unicode_sub_ascii_prompt_transformer() -> None:
    transformer = UnicodeSubstitutionConverter(0x00000)
    assert transformer.convert("test") == "\U00000074\U00000065\U00000073\U00000074"
