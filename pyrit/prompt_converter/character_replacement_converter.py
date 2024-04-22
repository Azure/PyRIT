# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from pyrit.models import PromptDataType
from pyrit.prompt_converter import PromptConverter, ConverterResult


class CharacterReplacementConverter(PromptConverter):
    """Converts a string by replacing chosen character with a new character of choice
    Args:
        char_to_replace (optional str): the character to replace (whitespaces by default)
        new_char (optional str): the new character to replace with (_ by default)
    """

    def __init__(self, char_to_replace: str = " ", new_char: str = "_") -> None:
        self.char_to_replace = char_to_replace
        self.new_char = new_char

    def convert(self, *, prompt: str, input_type: PromptDataType = "text") -> ConverterResult:
        """
        Simple converter that just replaces character in string with a chosen new character
        """
        if not self.input_supported(input_type):
            raise ValueError("Input type not supported")

        return ConverterResult(output_text=prompt.replace(self.char_to_replace, self.new_char), output_type="text")

    def input_supported(self, input_type: PromptDataType) -> bool:
        return input_type == "text"
