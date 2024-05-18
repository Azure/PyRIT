# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import random
import asyncio

from pyrit.models import PromptDataType
from pyrit.prompt_converter import PromptConverter, ConverterResult
from confusables import confusable_characters


class UnicodeConfusableConverter(PromptConverter):
    def __init__(self, deterministic: bool = False):
        """Set up a converter. The 'deterministic' argument is for unittesting only."""
        self.deterministic = deterministic

    async def convert_async(self, *, prompt: str, input_type: PromptDataType = "text") -> ConverterResult:
        """
        Converts the given prompts into things that look similar, but are actually different,
        using Unicode confusables -- e.g., replacing a Latin 'a' with a Cyrillic 'а'.

        This is sort of running UTR-39 in reverse, *introducing* confusables rather than
        removing them. (https://www.unicode.org/reports/tr39/tr39-1.html)

        Args:
            prompts (list[str]): The prompts to be converted.

        Returns:
            list[str]: The converted representations of the prompts.
        """
        if not self.input_supported(input_type):
            raise ValueError("Input type not supported")

        return_text = "".join(self._confusable(c) for c in prompt)
        await asyncio.sleep(0)
        return ConverterResult(output_text=return_text, output_type="text")

    def input_supported(self, input_type: PromptDataType) -> bool:
        return input_type == "text"

    def _confusable(self, char: str) -> str:
        """Pick a confusable character for the given character."""
        confusable_options = confusable_characters(char)
        if not confusable_options or char == " ":
            return char
        elif self.deterministic or len(confusable_options) == 1:
            return confusable_options[-1]
        else:
            return random.choice(confusable_options)
