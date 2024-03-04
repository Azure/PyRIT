# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from pyrit.prompt_converter import PromptConverter
from art import text2art


class AsciiArtConverter(PromptConverter):
    """Converts a string to ASCII art"""

    def __init__(self, font="rand"):
        self.font_value = font

    def convert(self, prompts: list[str]) -> list[str]:
        """
        Converter that uses art to convert strings to ASCII art.
        This can sometimes bypass LLM filters
        Args:
            prompts (list[str]): The prompts to be converted.
        Returns:
            list[str]: The converted prompts.
        """
        return [text2art(prompt, font=self.font_value) for prompt in prompts]
