# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from pyrit.prompt_converter import PromptConverter, ConverterResult
from pyrit.models import PromptDataType


class ZeroWidthConverter(PromptConverter):
    """
    A PromptConverter that injects zero-width spaces between characters
    in the provided text to bypass content safety mechanisms.
    """

    ZERO_WIDTH_SPACE = "\u200B"

    def input_supported(self, input_type: PromptDataType) -> bool:
        """
        Checks if the input type is supported by this converter.
        Supports only 'text' input type.

        Args:
            input_type (PromptDataType): The type of input to check (e.g., "text").

        Returns:
            bool: True if the input type is "text", otherwise False.
        """
        return input_type == "text"

    async def convert_async(self, *, prompt: str, input_type: PromptDataType = "text") -> ConverterResult:
        """
        Converts the given prompt by injecting zero-width spaces between each character.

        Args:
            prompt (str): The prompt to be converted.

        Returns:
            ConverterResult: The result containing the modified prompt.
        """
        if not self.input_supported(input_type):
            raise ValueError("Only 'text' input type is supported.")

        # Insert zero-width spaces between each character
        modified_text = self.ZERO_WIDTH_SPACE.join(prompt)

        return ConverterResult(output_text=modified_text, output_type="text")
