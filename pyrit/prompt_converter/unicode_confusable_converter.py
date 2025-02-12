# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging
import random
import re
from typing import Literal

from confusable_homoglyphs.confusables import is_confusable
from confusables import confusable_characters

from pyrit.models import PromptDataType
from pyrit.prompt_converter import ConverterResult, PromptConverter

logger = logging.getLogger(__name__)


class UnicodeConfusableConverter(PromptConverter):
    """
    A PromptConverter that applies substitutions to words in the prompt
    to test adversarial textual robustness by replacing characters with visually similar ones.
    """

    def __init__(
        self,
        *,
        source_package: Literal["confusable_homoglyphs", "confusables"] = "confusable_homoglyphs",
        deterministic: bool = False,
    ):
        """
        Initializes the UnicodeConfusableConverter.

        Args:
            source_package: The package to use for homoglyph generation. Can be either "confusable_homoglyphs"
                which can be found here: https://pypi.org/project/confusable-homoglyphs/ or "confusables" which can be
                found here: https://pypi.org/project/confusables/. "Confusable_homoglyphs" is used by default as it is
                more regularly maintained and up to date with the latest Unicode-provided confusables found here:
                https://www.unicode.org/Public/security/latest/confusables.txt. However, "confusables"
                provides additional methods of matching characters (not just Unicode list), so each character
                has more possible substitutions.
            deterministic: This argument is for unittesting only.
        """
        if source_package not in ["confusable_homoglyphs", "confusables"]:
            raise ValueError(
                f"Invalid source package: {source_package}. Please choose either 'confusable_homoglyphs' \
                or 'confusables'"
            )
        self._source_package = source_package
        self._deterministic = deterministic

    async def convert_async(self, *, prompt: str, input_type="text") -> ConverterResult:
        """
        Converts the given prompt by applying confusable substitutions. This leads to a prompt that looks similar,
        but is actually different (e.g., replacing a Latin 'a' with a Cyrillic 'а').

        Args:
            prompt (str): The prompt to be converted.
            input_type (str): The type of input (should be "text").
        Returns:
            ConverterResult: The result containing the prompt with confusable subsitutions applied.
        """
        if not self.input_supported(input_type):
            raise ValueError("Input type not supported")

        if self._source_package == "confusable_homoglyphs":
            converted_prompt = self._generate_perturbed_prompts(prompt)
        else:
            converted_prompt = "".join(self._confusable(c) for c in prompt)

        return ConverterResult(output_text=converted_prompt, output_type="text")

    def _get_homoglyph_variants(self, word: str) -> list:
        """
        Retrieves homoglyph variants for a given word using the "confusable_homoglyphs" package.

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
        Generates a perturbed prompt by substituting characters with their homoglyph variants using the
        "confusable_homoglyphs" package.

        Args:
            prompt (str): The original prompt.
        Returns:
            str: A perturbed prompt with character-level substitutions.
        """
        perturbed_words = []

        # Split the prompt into words and non-word tokens
        word_list = re.findall(r"\w+|\W+", prompt)

        for word in word_list:
            perturbed_chars = []
            for char in word:
                homoglyph_variants = self._get_homoglyph_variants(char)
                if homoglyph_variants:
                    # Randomly choose a homoglyph variant
                    variant = random.choice(homoglyph_variants) if not self._deterministic else homoglyph_variants[-1]
                    logger.debug(f"Replacing character '{char}' with '{variant}'")
                    perturbed_chars.append(variant)
                else:
                    perturbed_chars.append(char)
            perturbed_words.append("".join(perturbed_chars))

        # Join the perturbed words back into a string
        new_prompt = "".join(perturbed_words)
        logger.info(f"Final perturbed prompt: {new_prompt}")

        return new_prompt

    def _confusable(self, char: str) -> str:
        """
        Pick a confusable character for the given character using the "confusables" package.

        Args:
            char (str): The character to be replaced.
        Returns:
            str: The confusable character to replace the given character.
        """
        confusable_options = confusable_characters(char)
        if not confusable_options or char == " ":
            return char
        elif self._deterministic or len(confusable_options) == 1:
            return confusable_options[-1]
        else:
            return random.choice(confusable_options)

    def input_supported(self, input_type: PromptDataType) -> bool:
        return input_type == "text"

    def output_supported(self, output_type: PromptDataType) -> bool:
        return output_type == "text"
