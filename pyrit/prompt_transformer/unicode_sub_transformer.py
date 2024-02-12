# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from pyrit.prompt_transformer import PromptTransformer


class UnicodeSubstitutionTransformer(PromptTransformer):
    def __init__(self, start_value=0xE0000):
        self.startValue = start_value

    def transform(self, prompt: str) -> str:
        """
        Simple transformer that just encodes the prompt using any unicode starting point.
        Default is to use invisible flag emoji characters.
        """
        return "".join(chr(self.startValue + ord(ch)) for ch in prompt)
