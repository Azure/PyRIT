# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import abc
import asyncio
from dataclasses import dataclass
import re

from pyrit.models import PromptDataType, Identifier


@dataclass
class ConverterResult:
    output_text: str
    output_type: PromptDataType

    def __str__(self):
        return f"{self.output_type}: {self.output_text}"


class PromptConverter(abc.ABC, Identifier):
    """
    A prompt converter is responsible for converting prompts into a different representation.

    """

    @abc.abstractmethod
    async def convert_async(self, *, prompt: str, input_type: PromptDataType = "text") -> ConverterResult:
        """
        Converts the given prompts into a different representation

        Args:
            prompt: The prompt to be converted.

        Returns:
            str: The converted representation of the prompts.
        """
        pass

    @abc.abstractmethod
    def input_supported(self, input_type: PromptDataType) -> bool:
        """
        Checks if the input type is supported by the converter

        Args:
            input_type: The input type to check

        Returns:
            bool: True if the input type is supported, False otherwise
        """
        pass

    async def convert_tokens_async(
        self, *, prompt: str, input_type: PromptDataType = "text", start_token: str = "⟪", end_token: str = "⟫"
    ) -> ConverterResult:
        """
        Converts substrings within a prompt that are enclosed by specified start and end tokens. If there are no tokens
        present, the entire prompt is converted.

        Args:
            prompt (str): The input prompt containing text to be converted.
            input_type (str): The type of input data. Defaults to "text".
            start_token (str): The token indicating the start of a substring to be converted. Defaults to "⟪" which is
                relatively distinct.
            end_token (str): The token indicating the end of a substring to be converted. Defaults to "⟫" which is
                relatively distinct.

        Returns:
            str: The prompt with specified substrings converted.

        Raises:
            ValueError: If the input is inconsistent.
        """
        if input_type != "text" and (start_token in prompt or end_token in prompt):
            raise ValueError("Input type must be text when start or end tokens are present.")

        # Find all matches between start_token and end_token
        pattern = re.escape(start_token) + "(.*?)" + re.escape(end_token)
        matches = re.findall(pattern, prompt)

        if not matches:
            # No tokens found, convert the entire prompt
            return await self.convert_async(prompt=prompt, input_type=input_type)

        if prompt.count(start_token) != prompt.count(end_token):
            raise ValueError("Uneven number of start tokens and end tokens.")

        tasks = [self._replace_text_match(match) for match in matches]
        converted_parts = await asyncio.gather(*tasks)

        for original, converted in zip(matches, converted_parts):
            prompt = prompt.replace(f"{start_token}{original}{end_token}", converted.output_text, 1)

        return ConverterResult(output_text=prompt, output_type="text")

    async def _replace_text_match(self, match):
        result = await self.convert_async(prompt=match, input_type="text")
        return result

    def get_identifier(self):
        public_attributes = {}
        public_attributes["__type__"] = self.__class__.__name__
        public_attributes["__module__"] = self.__class__.__module__
        return public_attributes
