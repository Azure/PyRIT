# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging
import math
import random
import re
import string

from pyrit.models import PromptDataType
from pyrit.prompt_converter import ConverterResult, PromptConverter

# Use logger
logger = logging.getLogger(__name__)


class CharSwapGenerator(PromptConverter):
    """
    A PromptConverter that applies character swapping to words in the prompt
    to test adversarial textual robustness.
    """

    def __init__(self, *, max_iterations: int = 10, word_swap_ratio: float = 0.2):
        """
        Initializes the CharSwapConverter.

        Args:
            max_iterations (int): Number of times to generate perturbed prompts.
                The higher the number the higher the chance that words are different from the original prompt.
            word_swap_ratio (float): Percentage of words to perturb in the prompt per iteration.
        """
        super().__init__()

        # Ensure max_iterations is positive
        if max_iterations <= 0:
            raise ValueError("max_iterations must be greater than 0")

        # Ensure word_swap_ratio is between 0 and 1
        if not (0 < word_swap_ratio <= 1):
            raise ValueError("word_swap_ratio must be between 0 and 1 (exclusive of 0)")

        self.max_iterations = max_iterations
        self.word_swap_ratio = word_swap_ratio

    def input_supported(self, input_type: PromptDataType) -> bool:
        return input_type == "text"

    def output_supported(self, output_type: PromptDataType) -> bool:
        return output_type == "text"

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

    async def convert_async(self, *, prompt: str, input_type="text") -> ConverterResult:
        """
        Converts the given prompt by applying character swaps.

        Args:
            prompt (str): The prompt to be converted.
        Returns:
            ConverterResult: The result containing the perturbed prompts.
        """
        if not self.input_supported(input_type):
            raise ValueError("Input type not supported")

        # Tokenize the prompt into words and punctuation using regex
        words = re.findall(r"\w+|\S+", prompt)
        word_list_len = len(words)
        num_perturb_words = max(1, math.ceil(word_list_len * self.word_swap_ratio))

        # Copy the original word list for perturbation
        perturbed_word_list = words.copy()

        # Get random indices of words to undergo swapping
        random_words_idx = self._get_n_random(0, word_list_len, num_perturb_words)

        # Apply perturbation by swapping characters in the selected words
        for idx in random_words_idx:
            perturbed_word_list[idx] = self._perturb_word(perturbed_word_list[idx])

        # Join the perturbed words back into a prompt
        new_prompt = " ".join(perturbed_word_list)

        # Clean up spaces around punctuation
        output_text = re.sub(r'\s([?.!,\'"])', r"\1", new_prompt).strip()

        return ConverterResult(output_text=output_text, output_type="text")

    def _get_n_random(self, low: int, high: int, n: int) -> list:
        """
        Utility function to generate random indices.
        Words at these indices will be subjected to perturbation.
        """
        result = []
        try:
            result = random.sample(range(low, high), n)
        except ValueError:
            logger.debug(f"[CharSwapConverter] Sample size of {n} exceeds population size of {high - low}")
        return result
