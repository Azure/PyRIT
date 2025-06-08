# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import re
from typing import List, Optional, Union

from pyrit.prompt_converter.word_level_converter import WordLevelConverter


class StringJoinConverter(WordLevelConverter):
    """Converts text by joining its characters with the specified join value"""

    def __init__(
        self,
        *,
        join_value="-",
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
            join_value (str): The string used to join characters of each word.
            indices (Optional[List[int]]): Specific indices of words to convert.
            keywords (Optional[List[str]]): Keywords to select words for conversion.
            proportion (Optional[float]): Proportion of randomly selected words to convert [0.0-1.0].
            regex (Optional[Union[str, re.Pattern]]): Regex pattern to match words for conversion.
        """
        super().__init__(indices=indices, keywords=keywords, proportion=proportion, regex=regex)
        self.join_value = join_value

    async def convert_word_async(self, word: str) -> str:
        return self.join_value.join(word)
