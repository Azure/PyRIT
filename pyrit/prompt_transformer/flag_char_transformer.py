# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from pyrit.prompt_transformer import PromptTransformer


class FlagCharTransformer(PromptTransformer):
    def transform(self, prompt: str) -> str:
        """
        Simple transformer that just encodes the prompt as invisible flag emoji characters.
        """
        return "".join(chr(0xE0000 + ord(ch)) for ch in prompt)
