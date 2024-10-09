# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import re

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

    async def convert_async(self, *, prompt: str, input_type: PromptDataType = "text") -> ConverterResult:
        """
        Simple converter that just replaces character in string with a chosen new character

        Args:
            prompt (str): prompt to convert
            input_type (PromptDataType): type of input

        Returns: converted text as a ConverterResult object
        """
        if not self.input_supported(input_type):
            raise ValueError("Input type not supported")
        return ConverterResult(output_text=re.sub(self.old_value, self.new_value, prompt), output_type="text")

    def input_supported(self, input_type: PromptDataType) -> bool:
        return input_type == "text"
