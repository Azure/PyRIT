# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from pyrit.models import PromptDataType
from pyrit.prompt_converter import PromptConverter


class UnicodeSubstitutionConverter(PromptConverter):
    def __init__(self, *, start_value=0xE0000):
        self.startValue = start_value

    def convert(self, *, prompt: str, input_type: PromptDataType = "text") -> str:
        """
        Simple converter that just encodes the prompt using any unicode starting point.
        Default is to use invisible flag emoji characters.
        """
        if not self.is_supported(input_type):
            raise ValueError("Input type not supported")

        return "".join(chr(self.startValue + ord(ch)) for ch in prompt)

    def is_supported(self, input_type: PromptDataType) -> bool:
        return input_type == "text"
