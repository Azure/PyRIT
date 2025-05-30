# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import random
import re

from typing import List, Optional, Union

from pyrit.prompt_converter.word_level_converter import WordLevelConverter


class LeetspeakConverter(WordLevelConverter):
    """Converts a string to a leetspeak version."""

    def __init__(
        self,
        *,
        deterministic: bool = True,
        custom_substitutions: Optional[dict] = None,
        indices: Optional[List[int]] = None,
        keywords: Optional[List[str]] = None,
        proportion: Optional[float] = None,
        regex: Optional[Union[str, re.Pattern]] = None,
    ):
        """
        Initialize the converter with optional deterministic mode and custom substitutions.
        This class allows for selection of words to convert based on various criteria.
        Only one selection parameter may be provided at a time (indices, keywords, proportion, or regex).
        If no selection parameter is provided, all words will be converted.

        Args:
            deterministic (bool): If True, use the first substitution for each character.
                If False, randomly choose a substitution for each character.
            custom_substitutions (Optional[dict]): A dictionary of custom substitutions to override the defaults.
            indices (Optional[List[int]]): Specific indices of words to convert.
            keywords (Optional[List[str]]): Keywords to select words for conversion.
            proportion (Optional[float]): Proportion of randomly selected words to convert [0.0-1.0].
            regex (Optional[Union[str, re.Pattern]]): Regex pattern to match words for conversion.
        """
        super().__init__(indices=indices, keywords=keywords, proportion=proportion, regex=regex)

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
