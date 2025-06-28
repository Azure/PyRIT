# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from art import text2art

from pyrit.models import PromptDataType
from pyrit.prompt_converter import ConverterResult, PromptConverter


class AsciiArtConverter(PromptConverter):
    """
    Uses the `art` package to convert text into ASCII art.
    """

    def __init__(self, font="rand"):
        """
        Initializes the converter with a specified font.

        Args:
            font (str): The font to use for ASCII art. Defaults to "rand" which selects a random font.
        """
        self.font_value = font

    async def convert_async(self, *, prompt: str, input_type: PromptDataType = "text") -> ConverterResult:
        """Converts the given prompt into ASCII art."""
        if not self.input_supported(input_type):
            raise ValueError("Input type not supported")

        return ConverterResult(output_text=text2art(prompt, font=self.font_value), output_type="text")

    def input_supported(self, input_type: PromptDataType) -> bool:
        return input_type == "text"

    def output_supported(self, output_type: PromptDataType) -> bool:
        return output_type == "text"
