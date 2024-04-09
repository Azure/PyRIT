# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from pyrit.memory.memory_models import PromptDataType
from pyrit.prompt_converter import PromptConverter


class CapitalLettersConverter(PromptConverter):
    def convert(self, *, prompt: str, input_type: PromptDataType = "text") -> str:
        """
        Simple converter that converts the prompt to capital letters.
        """
        if not self.is_supported(input_type):
            raise ValueError("Input type not supported")

        output = prompt.upper()
        return output

    def is_supported(self, input_type: PromptDataType) -> bool:
        return input_type == "text"
