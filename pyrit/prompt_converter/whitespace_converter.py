# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from pyrit.models import PromptDataType
from pyrit.prompt_converter import PromptConverter, ConverterResult


class WhitespaceConverter(PromptConverter):
    """Converts a string by replacing whitespaces with character
    Args:
        character (optional str): the character to replace whitespaces with. _ by default
    """

    def __init__(self, character: str = "_") -> None:
        self.character = character

    def convert(self, *, prompt: str, input_type: PromptDataType = "text") -> ConverterResult:
        """
        Simple converter that just replaces whitespace with character
        """
        if not self.input_supported(input_type):
            raise ValueError("Input type not supported")

        return ConverterResult(output_text=prompt.replace(" ", self.character), output_type="text")

    def input_supported(self, input_type: PromptDataType) -> bool:
        return input_type == "text"
