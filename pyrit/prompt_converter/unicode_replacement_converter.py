# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

from typing import Optional

from pyrit.prompt_converter.text_selection_strategy import WordSelectionStrategy
from pyrit.prompt_converter.word_level_converter import WordLevelConverter


class UnicodeReplacementConverter(WordLevelConverter):
    """
    Converts a prompt to its unicode representation.
    """

    def __init__(
        self,
        *,
        encode_spaces: bool = False,
        word_selection_strategy: Optional[WordSelectionStrategy] = None,
    ):
        """
        Initializes the converter with the specified selection strategy.

        Args:
            encode_spaces (bool): If True, spaces in the prompt will be replaced with unicode representation.
            word_selection_strategy (Optional[WordSelectionStrategy]): Strategy for selecting which words to convert.
                If None, all words will be converted.
        """
        super().__init__(word_selection_strategy=word_selection_strategy)
        self.encode_spaces = encode_spaces

    async def convert_word_async(self, word: str) -> str:
        return "".join(f"\\u{ord(ch):04x}" for ch in word)

    def join_words(self, words: list[str]) -> str:
        if self.encode_spaces:
            return "\\u0020".join(words)
        return super().join_words(words)
