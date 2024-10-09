# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging
import random
import re

from confusable_homoglyphs.confusables import is_confusable


from pyrit.prompt_converter import PromptConverter, ConverterResult


logger = logging.getLogger(__name__)


class HomoglyphGenerator(PromptConverter):
    """
    A PromptConverter that applies homoglyph substitutions to words in the prompt
    to test adversarial textual robustness.
    """

    def __init__(self, *, max_iterations: int = 20):
        """
        Initializes the HomoglyphGenerator.
        Args:
            max_iterations (int): Maximum number of convert_async calls allowed.
        """
        super().__init__()
        self.max_iterations = max_iterations

    def input_supported(self, input_type) -> bool:
        """
        Checks if the input type is supported by the converter.
        """
        return input_type == "text"

    def _get_homoglyph_variants(self, word: str) -> list:
        """
        Retrieves homoglyph variants for a given word.
        Args:
            word (str): The word to find homoglyphs for.
        Returns:
            list: A list of homoglyph variants for the word.
        """
        try:
            # Check for confusable homoglyphs in the word
            confusables = is_confusable(word, greedy=True)
            if confusables:
                # Return a list of all homoglyph variants instead of only the first one
                return [homoglyph["c"] for item in confusables for homoglyph in item["homoglyphs"]]
        except UnicodeDecodeError:
            logger.error(f"Cannot process word '{word}' due to UnicodeDecodeError. Returning empty list.")
            return []

        # Default return if no homoglyphs are found
        return []

    def _generate_perturbed_prompts(self, prompt: str) -> str:
        """
        Generates a perturbed prompt by substituting characters with their homoglyph variants.
        Args:
            prompt (str): The original prompt.
        Returns:
            str: A perturbed prompt with character-level substitutions.
        """
        perturbed_words = []

        # Split the prompt into words and non-word tokens
        word_list = re.findall(r"\w+|\W+", prompt)

        for word in word_list:
            perturbed_word = ""
            for char in word:
                homoglyph_variants = self._get_homoglyph_variants(char)
                logger.info(f"Character: '{char}', Homoglyphs: {homoglyph_variants}")

                if homoglyph_variants:
                    # Randomly choose a homoglyph variant instead of always choosing the first one
                    variant = random.choice(homoglyph_variants)
                    logger.info(f"Replacing character '{char}' with '{variant}'")
                    perturbed_word += variant
                else:
                    perturbed_word += char
            perturbed_words.append(perturbed_word)

        # Join the perturbed words back into a string
        new_prompt = ''.join(perturbed_words)
        logger.info(f"Final perturbed prompt: {new_prompt}")

        return new_prompt

    async def convert_async(self, *, prompt: str, input_type="text") -> ConverterResult:
        """
        Converts the given prompt by applying homoglyph substitutions.
        Args:
            prompt (str): The prompt to be converted.
            input_type (str): The type of input (should be "text").
        Returns:
            ConverterResult: The result containing the perturbed prompts.
        """
        if not self.input_supported(input_type):
            raise ValueError("Input type not supported")

        perturbed_prompt = self._generate_perturbed_prompts(prompt)

        return ConverterResult(output_text=perturbed_prompt, output_type="text")
