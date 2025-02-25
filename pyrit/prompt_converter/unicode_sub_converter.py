# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from pyrit.models import PromptDataType
from pyrit.prompt_converter import ConverterResult, PromptConverter


class UnicodeSubstitutionConverter(PromptConverter):
    def __init__(self, *, start_value=0xE0000):
        self.startValue = start_value

    async def convert_async(self, *, prompt: str, input_type: PromptDataType = "text") -> ConverterResult:
        """
        Simple converter that just encodes the prompt using any unicode starting point.
        Default is to use invisible flag emoji characters.
        """
        if not self.input_supported(input_type):
            raise ValueError("Input type not supported")

        ret_text = "".join(chr(self.startValue + ord(ch)) for ch in prompt)
        return ConverterResult(output_text=ret_text, output_type="text")

    def input_supported(self, input_type: PromptDataType) -> bool:
        return input_type == "text"

    def output_supported(self, output_type: PromptDataType) -> bool:
        return output_type == "text"
