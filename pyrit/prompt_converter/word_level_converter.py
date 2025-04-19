# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import abc

from pyrit.common.utils import select_word_indices
from pyrit.models.literals import PromptDataType
from pyrit.prompt_converter import PromptConverter
from pyrit.prompt_converter.prompt_converter import ConverterResult


class WordLevelConverter(PromptConverter):
    """
    Base class for word-level converters. Designed to convert text by processing each word individually.

    Word selection is based on the `mode` and `mode_kwargs` parameters.
    The `mode` parameter determines how words are selected for conversion.
    The `mode_kwargs` parameter allows for additional configuration options specific to the selected mode.
    Please refer to the `select_word_indices` function for more details on how to use these parameters.

    Note:
        The `convert_word_async` method is an abstract method that must be implemented by subclasses.
        It defines the conversion logic for each word.
    """

    def __init__(self, mode: str = "all", **mode_kwargs):
        self.mode = mode
        self.mode_kwargs = mode_kwargs

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
        selected_indices = select_word_indices(words=words, mode=self.mode, **self.mode_kwargs)

        # Convert only selected words
        for idx in selected_indices:
            words[idx] = await self.convert_word_async(words[idx])

        return ConverterResult(output_text=self.join_words(words), output_type="text")

    def input_supported(self, input_type: PromptDataType) -> bool:
        return input_type == "text"

    def output_supported(self, output_type: PromptDataType) -> bool:
        return output_type == "text"
