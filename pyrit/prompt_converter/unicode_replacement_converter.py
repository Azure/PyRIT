# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import re
from typing import List, Optional, Union

from pyrit.prompt_converter.word_level_converter import WordLevelConverter


class UnicodeReplacementConverter(WordLevelConverter):
    """Simple converter that returns the unicode representation of the prompt."""

    def __init__(
        self,
        *,
        encode_spaces: bool = False,
        indices: Optional[List[int]] = None,
        keywords: Optional[List[str]] = None,
        proportion: Optional[float] = None,
        regex: Optional[Union[str, re.Pattern]] = None,
    ):
        """
        Initialize the converter.
        This class allows for selection of words to convert based on various criteria.
        Only one selection parameter may be provided at a time (indices, keywords, proportion, or regex).
        If no selection parameter is provided, all words will be converted.

        Args:
            encode_spaces (bool): If True, spaces in the prompt will be replaced with unicode representation.
            indices (Optional[List[int]]): Specific indices of words to convert.
            keywords (Optional[List[str]]): Keywords to select words for conversion.
            proportion (Optional[float]): Proportion of randomly selected words to convert [0.0-1.0].
            regex (Optional[Union[str, re.Pattern]]): Regex pattern to match words for conversion.
        """
        super().__init__(indices=indices, keywords=keywords, proportion=proportion, regex=regex)
        self.encode_spaces = encode_spaces

    async def convert_word_async(self, word: str) -> str:
        return "".join(f"\\u{ord(ch):04x}" for ch in word)

    def join_words(self, words: list[str]) -> str:
        if self.encode_spaces:
            return "\\u0020".join(words)
        return super().join_words(words)
