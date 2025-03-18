# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from pyrit.models import PromptDataType
from pyrit.prompt_converter import ConverterResult, PromptConverter


class UnicodeReplacementConverter(PromptConverter):

    async def convert_async(self, *, prompt: str, input_type: PromptDataType = "text") -> ConverterResult:
        """
        Simple converter that returnst the unicode representation of the prompt.
        """
        if not self.input_supported(input_type):
            raise ValueError("Input type not supported")

        ret_text = "".join(f"\\u{ord(ch):04x}" for ch in prompt)
        return ConverterResult(output_text=ret_text, output_type="text")

    def input_supported(self, input_type: PromptDataType) -> bool:
        return input_type == "text"

    def output_supported(self, output_type: PromptDataType) -> bool:
        return output_type == "text"