# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from pyrit.models import PromptDataType
from pyrit.prompt_converter import ConverterResult, PromptConverter


class FlipConverter(PromptConverter):

    async def convert_async(self, *, prompt: str, input_type: PromptDataType = "text") -> ConverterResult:
        """
        Simple converter that flips the prompt. "hello me" would be "em olleh"
        """
        if not self.input_supported(input_type):
            raise ValueError("Input type not supported")

        rev_prompt = prompt[::-1]

        return ConverterResult(output_text=rev_prompt, output_type="text")

    def input_supported(self, input_type: PromptDataType) -> bool:
        return input_type == "text"

    def output_supported(self, output_type: PromptDataType) -> bool:
        return output_type == "text"
