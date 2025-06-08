# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging
import random
import re
from typing import List, Optional, Union

from pyrit.prompt_converter.word_level_converter import WordLevelConverter

# Unicode combining characters for Zalgo effect (U+0300–U+036F)
ZALGO_MARKS = [chr(code) for code in range(0x0300, 0x036F + 1)]
# Setting a max intensity so people don't do anything unreasonable
MAX_INTENSITY = 100
logger = logging.getLogger(__name__)


class ZalgoConverter(WordLevelConverter):
    """Converts text into cursed Zalgo text using combining Unicode marks."""

    def __init__(
        self,
        *,
        intensity: int = 10,
        seed: Optional[int] = None,
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
            intensity (int): Number of combining marks per character (higher = more cursed). Default is 10.
            seed (Optional[int]): Optional seed for reproducible output.
            indices (Optional[List[int]]): Specific indices of words to convert.
            keywords (Optional[List[str]]): Keywords to select words for conversion.
            proportion (Optional[float]): Proportion of randomly selected words to convert [0.0-1.0].
            regex (Optional[Union[str, re.Pattern]]): Regex pattern to match words for conversion.
        """
        super().__init__(indices=indices, keywords=keywords, proportion=proportion, regex=regex)
        self._intensity = self._normalize_intensity(intensity)
        self._seed = seed

    def _normalize_intensity(self, intensity: int) -> int:
        try:
            intensity = int(intensity)
        except (TypeError, ValueError):
            raise ValueError(f"Invalid intensity value: {intensity!r} (must be an integer)")

        normalized_intensity = max(0, min(intensity, MAX_INTENSITY))
        if intensity != normalized_intensity:
            logger.warning(
                f"ZalgoConverter supports intensity between 0 and {MAX_INTENSITY}, "
                f"but received a value of {intensity}. Normalizing to {normalized_intensity}."
            )
        return normalized_intensity

    async def convert_word_async(self, word: str) -> str:
        if self._intensity <= 0:
            return word

        def glitch(char: str) -> str:
            return char + "".join(random.choice(ZALGO_MARKS) for _ in range(random.randint(1, self._intensity)))

        return "".join(glitch(c) if c.isalnum() else c for c in word)

    def validate_input(self, prompt: str) -> None:
        # Initialize the random seed before processing any words
        if self._seed is not None:
            random.seed(self._seed)
