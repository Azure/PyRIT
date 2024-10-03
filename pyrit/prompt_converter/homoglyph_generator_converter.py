# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging
import re

import homoglyphs as hg

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
            max_iterations (int): Maximum number of perturbed prompts to generate.
        """
        super().__init__()
        self.max_iterations = max_iterations
        self.homoglyphs = hg.Homoglyphs(
            strategy=hg.STRATEGY_LOAD, ascii_strategy=hg.STRATEGY_REMOVE
        )

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
            variants = self.homoglyphs.to_ascii(word)
            # Exclude the original word to avoid duplicates
            return [variant for variant in variants if variant != word]
        except UnicodeDecodeError:
            logger.error(f"Cannot process word '{word}'. Skipping.")
            return []

    def _generate_perturbed_prompts(self, prompt: str) -> list:
        """
        Generates perturbed prompts by substituting words with their homoglyph variants.
        Args:
            prompt (str): The original prompt.
        Returns:
            list: A list of perturbed prompts.
        """
        result_list = []
        count = 0
        word_list = re.findall(r'\w+|\S+', prompt)
        word_list_len = len(word_list)

        for idx in range(word_list_len):
            if count >= self.max_iterations:
                break
            homoglyph_variants = self._get_homoglyph_variants(word_list[idx])

            if homoglyph_variants:
                for variant in homoglyph_variants:
                    if count >= self.max_iterations:
                        break
                    perturbed_word_list = word_list.copy()
                    perturbed_word_list[idx] = variant
                    # Detokenize using join and remove extra spaces before punctuation
                    new_prompt = " ".join(perturbed_word_list)
                    new_prompt = re.sub(r'\s([?.!,\'"])', r'\1', new_prompt).strip()
                    result_list.append(new_prompt)
                    count += 1
        return result_list

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

        perturbed_prompts = self._generate_perturbed_prompts(prompt)

        # Combine the perturbed prompts into a single output
        output_text = "\n".join(perturbed_prompts)
        return ConverterResult(output_text=output_text, output_type="text")
