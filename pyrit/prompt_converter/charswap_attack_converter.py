# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import random
import string

from pyrit.prompt_converter.word_level_converter import WordLevelConverter


class CharSwapConverter(WordLevelConverter):
    """Applies character swapping to words in the prompt to test adversarial textual robustness."""

    def __init__(self, *, max_iterations: int = 10):
        """
        Args:
            max_iterations (int): Number of times to generate perturbed prompts.
                The higher the number the higher the chance that words are different from the original prompt.
        """
        super().__init__()
        self.select_random(0.2)

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
