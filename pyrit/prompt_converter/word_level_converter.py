# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import abc
import logging
import re
from typing import List, TypeVar, Union, final

from pyrit.common.utils import get_random_indices
from pyrit.models.literals import PromptDataType
from pyrit.prompt_converter import PromptConverter
from pyrit.prompt_converter.prompt_converter import ConverterResult

logger = logging.getLogger(__name__)

# Define a generic type variable for self-returning methods
T = TypeVar("T", bound="WordLevelConverter")


class WordLevelConverter(PromptConverter):
    """
    Base class for word-level converters. Designed to convert text by processing each word individually.

    Word selection is based on configuration methods provided by the class.
    These methods define how words are selected for conversion.

    Note:
        The `convert_word_async` method is an abstract method that must be implemented by subclasses.
        It defines the conversion logic for each word.
    """

    def __init__(self):
        self._selection_mode = "all"
        self._selection_indices = []
        self._selection_keywords = []
        self._selection_proportion = 0.5
        self._selection_regex = r"."

    @abc.abstractmethod
    async def convert_word_async(self, word: str) -> str:
        pass

    @final
    def select_all(self: T) -> T:
        """Configure the converter to convert all words."""
        self._selection_mode = "all"
        return self

    @final
    def select_custom(self: T, indices: List[int] = []) -> T:
        """Configure the converter to only convert words at specific indices."""
        self._selection_mode = "custom"
        self._selection_indices = indices
        return self

    @final
    def select_keywords(self: T, keywords: List[str] = []) -> T:
        """Configure the converter to only convert words matching specific keywords."""
        self._selection_mode = "keywords"
        self._selection_keywords = keywords
        return self

    @final
    def select_random(self: T, proportion: float = 0.5) -> T:
        """Configure the converter to only convert a random selection of words based on a proportion."""
        self._selection_mode = "random"
        self._selection_proportion = proportion
        return self

    @final
    def select_regex(self: T, pattern: Union[str, re.Pattern] = r".") -> T:
        """Configure the converter to only convert words matching a regex pattern."""
        self._selection_mode = "regex"
        self._selection_regex = pattern
        return self

    @final
    def _select_word_indices(self, words: List[str]) -> List[int]:
        """
        Select indices from a list of words based on the current selection configuration.

        Args:
            words (List[str]): A list of words to select from.

        Returns:
            List[int]: Indices of selected words.
        """
        if not words:
            return []

        mode = self._selection_mode

        match mode:
            case "all":
                return list(range(len(words)))
            case "keywords":
                return [i for i, word in enumerate(words) if word in self._selection_keywords]
            case "random":
                return get_random_indices(start=0, size=len(words), proportion=self._selection_proportion)
            case "regex":
                return [i for i, word in enumerate(words) if re.search(self._selection_regex, word)]
            case "custom":
                custom_indices = self._selection_indices or []
                valid_indices = [i for i in custom_indices if 0 <= i < len(words)]
                invalid_indices = [i for i in custom_indices if i < 0 or i >= len(words)]
                if invalid_indices:
                    raise ValueError(
                        f"Invalid indices {invalid_indices} provided for custom selection. "
                        f"Valid range is 0 to {len(words) - 1}."
                    )
                return valid_indices
            case _:
                return list(range(len(words)))

    def validate_input(self, prompt: str) -> None:
        """Validate the input before processing (can be overridden by subclasses)"""
        pass

    def join_words(self, words: list[str]) -> str:
        """Provide a way for subclasses to override the default behavior of joining words."""
        return " ".join(words)

    async def convert_async(self, *, prompt: str, input_type: PromptDataType = "text") -> ConverterResult:
        if prompt is None:
            raise TypeError("Prompt cannot be None")

        if input_type != "text":
            raise ValueError(f"Input type {input_type} not supported")

        self.validate_input(prompt=prompt)

        words = prompt.split(" ")  # split by spaces only, preserving other whitespace
        selected_indices = self._select_word_indices(words=words)

        # Convert only selected words
        for idx in selected_indices:
            words[idx] = await self.convert_word_async(words[idx])

        return ConverterResult(output_text=self.join_words(words), output_type="text")

    def input_supported(self, input_type: PromptDataType) -> bool:
        return input_type == "text"

    def output_supported(self, output_type: PromptDataType) -> bool:
        return output_type == "text"
