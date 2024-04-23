# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from pyrit.models import PromptDataType
from pyrit.prompt_converter import PromptConverter, ConverterResult


class SearchReplaceConverter(PromptConverter):
    """Converts a string by replacing chosen phrase with a new phrase of choice
    Args:
        old_value (str): the phrase to replace
        new_value (str): the new phrase to replace with
    """

    def __init__(self, old_value: str, new_value: str) -> None:
        self.old_value = old_value
        self.new_value = new_value

    def convert(self, *, prompt: str, input_type: PromptDataType = "text") -> ConverterResult:
        """
        Simple converter that just replaces character in string with a chosen new character
        """
        if not self.input_supported(input_type):
            raise ValueError("Input type not supported")

        return ConverterResult(output_text=prompt.replace(self.char_to_replace, self.new_char), output_type="text")

    def input_supported(self, input_type: PromptDataType) -> bool:
        return input_type == "text"
