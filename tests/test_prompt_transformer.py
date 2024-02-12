# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from pyrit.prompt_transformer import Base64Transformer, NoOpTransformer, UnicodeSubstitutionTransformer


def test_prompt_transformer() -> None:
    transformer = NoOpTransformer()
    assert transformer.transform("test") == "test"


def test_base64_prompt_transformer() -> None:
    transformer = Base64Transformer()
    assert transformer.transform("test") == "dGVzdA=="


def test_unicode_sub_default_prompt_transformer() -> None:
    transformer = UnicodeSubstitutionTransformer()
    assert transformer.transform("test") == "\U000e0074\U000e0065\U000e0073\U000e0074"


def test_unicode_sub_ascii_prompt_transformer() -> None:
    transformer = UnicodeSubstitutionTransformer(0x00000)
    assert transformer.transform("test") == "\U00000074\U00000065\U00000073\U00000074"
