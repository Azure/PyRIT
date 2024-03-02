# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from pyrit.prompt_converter import PromptConverter
from art import text2art


class AsciiArtConverter(PromptConverter):
    "Converts a string to ASCII art"

    def __init__(self, font="rand"):
        self.font_value = font

    def convert(self, prompt: str) -> str:
        """
        Converter that uses art to convert a string to ASCII art
        This can sometimes bypass LLM filters
        Args:
            prompt (str): The prompt to be converted.
        Returns:
            str: The converted prompt.
        """
        return text2art(prompt, font=self.font_value)
