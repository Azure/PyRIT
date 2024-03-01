# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from pyrit.prompt_converter import PromptConverter


class UnicodeSubstitutionConverter(PromptConverter):
    def __init__(self, *, start_value=0xE0000, include_original=False):
        self.startValue = start_value
        self.include_original = include_original

    def convert(self, prompts: list[str]) -> list[str]:
        """
        Simple converter that just encodes the prompt using any unicode starting point.
        Default is to use invisible flag emoji characters.
        """
        ret_list = prompts[:] if self.include_original else []

        for prompt in prompts:
            ret_list.append("".join(chr(self.startValue + ord(ch)) for ch in prompt))

        return ret_list
