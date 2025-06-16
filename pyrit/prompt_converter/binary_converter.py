# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

import re
from enum import Enum
from typing import List, Optional, Union

from pyrit.prompt_converter.word_level_converter import WordLevelConverter


class BinaryConverter(WordLevelConverter):
    """
    Transforms input text into its binary representation with configurable bits per character (8, 16, or 32).
    """

    class BitsPerChar(Enum):
        BITS_8 = 8
        BITS_16 = 16
        BITS_32 = 32

    def __init__(
        self,
        *,
        bits_per_char: BinaryConverter.BitsPerChar = BitsPerChar.BITS_16,
        indices: Optional[List[int]] = None,
        keywords: Optional[List[str]] = None,
        proportion: Optional[float] = None,
        regex: Optional[Union[str, re.Pattern]] = None,
    ):
        """
        Initializes the converter with the specified bits per character and selection parameters.

        This class allows for selection of words to convert based on various criteria.
        Only one selection parameter may be provided at a time (indices, keywords, proportion, or regex).
        If no selection parameter is provided, all words will be converted.

        Args:
            bits_per_char (BinaryConverter.BitsPerChar): Number of bits to use for each character (8, 16, or 32).
                Default is 16 bits.
            indices (Optional[List[int]]): Specific indices of words to convert.
            keywords (Optional[List[str]]): Keywords to select words for conversion.
            proportion (Optional[float]): Proportion of randomly selected words to convert [0.0-1.0].
            regex (Optional[Union[str, re.Pattern]]): Regex pattern to match words for conversion.
        """
        super().__init__(indices=indices, keywords=keywords, proportion=proportion, regex=regex)

        if not isinstance(bits_per_char, BinaryConverter.BitsPerChar):
            raise TypeError("bits_per_char must be an instance of BinaryConverter.BitsPerChar Enum.")
        self.bits_per_char = bits_per_char

    def validate_input(self, prompt):
        # Check if bits_per_char is sufficient for the characters in the prompt
        bits = self.bits_per_char.value
        max_code_point = max((ord(char) for char in prompt), default=0)
        min_bits_required = max_code_point.bit_length()
        if bits < min_bits_required:
            raise ValueError(
                f"bits_per_char={bits} is too small for the characters in the prompt. "
                f"Minimum required bits: {min_bits_required}."
            )

    async def convert_word_async(self, word: str) -> str:
        bits = self.bits_per_char.value
        # Convert each character in the word to its binary representation
        return " ".join(format(ord(char), f"0{bits}b") for char in word)

    def join_words(self, words: list[str]) -> str:
        """Join the converted words with the binary representation of a space."""
        space_binary = format(ord(" "), f"0{self.bits_per_char.value}b")
        return f" {space_binary} ".join(words)
