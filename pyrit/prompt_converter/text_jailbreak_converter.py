# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from pyrit.datasets import TextJailBreak
from pyrit.models import PromptDataType
from pyrit.prompt_converter import ConverterResult, PromptConverter


class TextJailBreakConverter(PromptConverter):

    def __init__(self, *, jail_break: TextJailBreak):
        if jail_break is None:
            raise TypeError("jail_break cannot be None")
        self.jail_break_template = jail_break

    async def convert_async(self, *, prompt: str, input_type: PromptDataType = "text") -> ConverterResult:
        """
        Simple converter that uses a jailbreak template to create a prompt
        """
        if not self.input_supported(input_type):
            raise ValueError("Input type not supported")

        jailbreak_prompt = self.jail_break_template.get_jailbreak(prompt=prompt)
        return ConverterResult(output_text=jailbreak_prompt, output_type="text")

    def input_supported(self, input_type: PromptDataType) -> bool:
        return input_type == "text"

    def output_supported(self, output_type: PromptDataType) -> bool:
        return output_type == "text"
