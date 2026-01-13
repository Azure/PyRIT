# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

from pyrit.models import PromptDataType
from pyrit.prompt_converter.prompt_converter import ConverterResult, PromptConverter


class FlipConverter(PromptConverter):
    """
    Flips the input text prompt. For example, "hello me" would be converted to "em olleh".
    """

    SUPPORTED_INPUT_TYPES = ("text",)
    SUPPORTED_OUTPUT_TYPES = ("text",)

    async def convert_async(self, *, prompt: str, input_type: PromptDataType = "text") -> ConverterResult:
        """Converts the given prompt by reversing the text."""
        if not self.input_supported(input_type):
            raise ValueError("Input type not supported")

        rev_prompt = prompt[::-1]

        return ConverterResult(output_text=rev_prompt, output_type="text")
