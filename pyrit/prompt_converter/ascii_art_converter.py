# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from pyrit.models import PromptDataType
from pyrit.prompt_converter import PromptConverter, ConverterResult
from art import text2art


class AsciiArtConverter(PromptConverter):
    """Converts a string to ASCII art"""

    def __init__(self, font="rand"):
        self.font_value = font

    def convert(self, *, prompt: str, input_type: PromptDataType = "text") -> ConverterResult:
        """
        Converter that uses art to convert strings to ASCII art.
        This can sometimes bypass LLM filters

        Args:
            prompt (str): The prompt to be converted.
        Returns:
            str: The converted prompt.
        """
        if not self.input_supported(input_type):
            raise ValueError("Input type not supported")

        return ConverterResult(output_text=text2art(prompt, font=self.font_value), output_type="text")

    def input_supported(self, input_type: PromptDataType) -> bool:
        return input_type == "text"
