# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from pyrit.prompt_transformer import PromptTransformer
from pyrit.prompt_transformer import Base64Transformer



def test_prompt_transformer() -> None:
    transformer = PromptTransformer()
    assert transformer.transform("test") == "test"

def test_base64_prompt_transformer() -> None:
    transformer = Base64Transformer()
    assert transformer.transform("test") == 'dGVzdA=='