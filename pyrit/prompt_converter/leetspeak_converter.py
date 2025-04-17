# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import random

from pyrit.prompt_converter.word_level_converter import WordLevelConverter


class LeetspeakConverter(WordLevelConverter):
    """Converts a string to a leetspeak version."""

    def __init__(
        self, *, deterministic: bool = True, custom_substitutions: dict = None, mode: str = "all", **mode_kwargs
    ):
        """
        Initialize the converter with optional deterministic mode and custom substitutions.

        Args:
            deterministic (bool): If True, use the first substitution for each character.
                If False, randomly choose a substitution for each character.
            custom_substitutions (dict, Optional): A dictionary of custom substitutions to override the defaults.
        """
        super().__init__(mode=mode, **mode_kwargs)

        default_substitutions = {
            "a": ["4", "@", "/\\", "@", "^", "/-\\"],
            "b": ["8", "6", "13", "|3", "/3", "!3"],
            "c": ["(", "[", "<", "{"],
            "e": ["3"],
            "g": ["9"],
            "i": ["1", "!"],
            "l": ["1", "|"],
            "o": ["0"],
            "s": ["5", "$"],
            "t": ["7"],
            "z": ["2"],
        }

        # Use custom substitutions if provided, otherwise default to the standard ones
        self._leet_substitutions = custom_substitutions if custom_substitutions else default_substitutions
        self._deterministic = deterministic

    async def convert_word_async(self, word: str) -> str:
        converted_word = []
        for char in word:
            lower_char = char.lower()
            if lower_char in self._leet_substitutions:
                if self._deterministic:
                    # Use the first substitution for deterministic mode
                    converted_word.append(self._leet_substitutions[lower_char][0])
                else:
                    # Randomly select a substitution for each character
                    converted_word.append(random.choice(self._leet_substitutions[lower_char]))
            else:
                # If character not in substitutions, keep it as is
                converted_word.append(char)
        return "".join(converted_word)
