# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from pyrit.models import PromptDataType
from pyrit.prompt_converter import ConverterResult, PromptConverter


class TextToHexConverter(PromptConverter):

    async def convert_async(self, *, prompt: str, input_type: PromptDataType = "text") -> ConverterResult:
        """
        Converts text to a hexadecimal encoded utf-8 string.
        """
        hex_representation = ""

        if not self.input_supported(input_type):
            raise ValueError("Input type not supported")

        hex_representation += prompt.encode("utf-8").hex().upper()

        return ConverterResult(output_text=hex_representation, output_type="text")

    def input_supported(self, input_type: PromptDataType) -> bool:
        return input_type == "text"

    def output_supported(self, output_type: PromptDataType) -> bool:
        return output_type == "text"
