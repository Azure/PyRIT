# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import codecs

from pyrit.models import PromptDataType
from pyrit.prompt_converter import ConverterResult, PromptConverter


class ROT13Converter(PromptConverter):

    async def convert_async(self, *, prompt: str, input_type: PromptDataType = "text") -> ConverterResult:
        """
        Simple converter that just ROT13 encodes the prompts
        """
        if not self.input_supported(input_type):
            raise ValueError("Input type not supported")

        result = ConverterResult(output_text=codecs.encode(prompt, "rot13"), output_type="text")
        return result

    def input_supported(self, input_type: PromptDataType) -> bool:
        return input_type == "text"

    def output_supported(self, output_type: PromptDataType) -> bool:
        return output_type == "text"
