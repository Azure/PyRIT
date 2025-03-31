# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

from enum import Enum

from pyrit.prompt_converter.word_level_converter import WordLevelConverter


class BinaryConverter(WordLevelConverter):
    """Transforms input text into its binary representation with configurable bits per character (8, 16, or 32)"""

    class BitsPerChar(Enum):
        BITS_8 = 8
        BITS_16 = 16
        BITS_32 = 32

    def __init__(
        self, bits_per_char: BinaryConverter.BitsPerChar = BitsPerChar.BITS_16, mode: str = "all", **mode_kwargs
    ):
        super().__init__(mode=mode, **mode_kwargs)

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
        return format(ord(word), f"0{bits}b")
