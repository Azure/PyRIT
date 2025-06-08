# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import random
import re
import string
from typing import List, Optional, Union

from pyrit.prompt_converter.word_level_converter import WordLevelConverter


class CharSwapConverter(WordLevelConverter):
    """Applies character swapping to words in the prompt to test adversarial textual robustness."""

    def __init__(
        self,
        *,
        max_iterations: int = 10,
        indices: Optional[List[int]] = None,
        keywords: Optional[List[str]] = None,
        proportion: Optional[float] = 0.2,
        regex: Optional[Union[str, re.Pattern]] = None,
    ):
        """
        Initialize the converter.
        This class allows for selection of words to convert based on various criteria.
        Only one selection parameter may be provided at a time (indices, keywords, proportion, or regex).
        By default, proportion is set to 0.2, meaning 20% of randomly selected words will be perturbed.

        Args:
            max_iterations (int): Number of times to generate perturbed prompts.
                The higher the number the higher the chance that words are different from the original prompt.
            indices (Optional[List[int]]): Specific indices of words to convert.
            keywords (Optional[List[str]]): Keywords to select words for conversion.
            proportion (Optional[float]): Proportion of randomly selected words to convert [0.0-1.0].
            regex (Optional[Union[str, re.Pattern]]): Regex pattern to match words for conversion.
        """
        super().__init__(indices=indices, keywords=keywords, proportion=proportion, regex=regex)

        # Ensure max_iterations is positive
        if max_iterations <= 0:
            raise ValueError("max_iterations must be greater than 0")

        self.max_iterations = max_iterations

    async def convert_word_async(self, word: str) -> str:
        return self._perturb_word(word)

    def _perturb_word(self, word: str) -> str:
        """
        Perturb a word by swapping two adjacent characters.
        Args:
            word (str): The word to perturb.
        Returns:
            str: The perturbed word with swapped characters.
        """
        if word not in string.punctuation and len(word) > 3:
            idx1 = random.randint(1, len(word) - 2)
            idx_elements = list(word)
            # Swap characters
            idx_elements[idx1], idx_elements[idx1 + 1] = (
                idx_elements[idx1 + 1],
                idx_elements[idx1],
            )
            return "".join(idx_elements)
        return word
