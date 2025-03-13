# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import random
import re

from pyrit.models import PromptDataType
from pyrit.prompt_converter import ConverterResult, PromptConverter


class SearchReplaceConverter(PromptConverter):
    """Converts a string by replacing chosen phrase with a new phrase of choice

    Args:
        pattern (str): the regex pattern to replace
        replace (str): the new phrase to replace with, can be a list and a random element is chosen
        regex_flags (int): regex flags to use for the replacement
    """

    def __init__(self, pattern: str, replace: str | list[str], regex_flags=0) -> None:
        self.pattern = pattern
        self.replace_list = [replace] if isinstance(replace, str) else replace

        self.regex_flags = regex_flags

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

        replace = random.choice(self.replace_list)

        return ConverterResult(
            output_text=re.sub(self.pattern, replace, prompt, flags=self.regex_flags), output_type="text"
        )

    def input_supported(self, input_type: PromptDataType) -> bool:
        return input_type == "text"

    def output_supported(self, output_type: PromptDataType) -> bool:
        return output_type == "text"
