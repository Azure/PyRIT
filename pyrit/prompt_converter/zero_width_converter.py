# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from pyrit.models import PromptDataType
from pyrit.prompt_converter import ConverterResult, PromptConverter


class ZeroWidthConverter(PromptConverter):
    """
    A PromptConverter that injects zero-width spaces between characters
    in the provided text to bypass content safety mechanisms.
    """

    ZERO_WIDTH_SPACE = "\u200b"

    def input_supported(self, input_type: PromptDataType) -> bool:
        return input_type == "text"

    def output_supported(self, output_type: PromptDataType) -> bool:
        return output_type == "text"

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
