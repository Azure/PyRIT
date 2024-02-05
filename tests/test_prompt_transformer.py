# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from pyrit.prompt_transformer import Base64Transformer, NoOpTransformer


def test_prompt_transformer() -> None:
    transformer = NoOpTransformer()
    assert transformer.transform("test") == "test"


def test_base64_prompt_transformer() -> None:
    transformer = Base64Transformer()
    assert transformer.transform("test") == "dGVzdA=="
