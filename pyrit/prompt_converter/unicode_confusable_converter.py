# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import random
from pyrit.prompt_converter import PromptConverter
from confusables import confusable_characters


class UnicodeConfusableConverter(PromptConverter):
    def __init__(self, deterministic: bool = False):
        """Set up a converter. The 'deterministic' argument is for unittesting only."""
        self.deterministic = deterministic

    def convert(self, prompts: list[str]) -> list[str]:
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
        return ["".join(self._confusable(c) for c in prompt) for prompt in prompts]

    def is_one_to_one_converter(self) -> bool:
        return True

    def _confusable(self, char: str) -> str:
        """Pick a confusable character for the given character."""
        if char == " ":
            return char

        confusable_options = confusable_characters(char)
        if len(confusable_options) < 2 or self.deterministic:
            return confusable_options[-1]
        else:
            return random.choice(confusable_options)
