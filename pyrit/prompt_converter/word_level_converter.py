# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import abc
import re
from typing import List, Optional, Union

from pyrit.common.utils import get_random_indices
from pyrit.models.literals import PromptDataType
from pyrit.prompt_converter import PromptConverter
from pyrit.prompt_converter.prompt_converter import ConverterResult


class WordLevelConverter(PromptConverter):
    """
    Base class for word-level converters. Designed to convert text by processing each word individually.

    Note:
        The `convert_word_async` method is an abstract method that must be implemented by subclasses.
        It defines the conversion logic for each word.
    """

    def __init__(
        self,
        *,
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
            indices (Optional[List[int]]): Specific indices of words to convert.
            keywords (Optional[List[str]]): Keywords to select words for conversion.
            proportion (Optional[float]): Proportion of randomly selected words to convert [0.0-1.0].
            regex (Optional[Union[str, re.Pattern]]): Regex pattern to match words for conversion.
        """
        # Make sure at most one selection criteria is provided
        criteria_map = {"indices": indices, "keywords": keywords, "proportion": proportion, "regex": regex}
        provided_criteria = {name: value for name, value in criteria_map.items() if value is not None}

        if len(provided_criteria) > 1:
            raise ValueError("Only one selection criteria can be provided at a time")

        if provided_criteria:
            self._mode = list(provided_criteria.keys())[0]
        else:
            self._mode = "all"

        self._keywords = keywords or []
        self._indices = indices or []
        self._proportion = 1.0 if proportion is None else proportion
        self._regex = regex or ".*"

    def _select_word_indices(self, words: List[str]) -> List[int]:
        """Return indices of words to be converted based on the selection criteria."""
        if not words:
            return []

        match self._mode:
            case "all":
                return list(range(len(words)))
            case "keywords":
                return [i for i, word in enumerate(words) if word in self._keywords]
            case "proportion":
                return get_random_indices(start=0, size=len(words), proportion=self._proportion)
            case "regex":
                return [i for i, word in enumerate(words) if re.search(self._regex, word)]
            case "indices":
                valid_indices = [i for i in self._indices if 0 <= i < len(words)]
                invalid_indices = [i for i in self._indices if i < 0 or i >= len(words)]
                if invalid_indices:
                    raise ValueError(
                        f"Invalid indices {invalid_indices} provided for custom selection."
                        f" Valid range is 0 to {len(words) - 1}."
                    )
                return valid_indices
            case _:
                return list(range(len(words)))

    @abc.abstractmethod
    async def convert_word_async(self, word: str) -> str:
        pass

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
