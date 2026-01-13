# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

import urllib.parse

from pyrit.models import PromptDataType
from pyrit.prompt_converter.prompt_converter import ConverterResult, PromptConverter


class UrlConverter(PromptConverter):
    """
    Converts a prompt to a URL-encoded string.
    """

    SUPPORTED_INPUT_TYPES = ("text",)
    SUPPORTED_OUTPUT_TYPES = ("text",)

    async def convert_async(self, *, prompt: str, input_type: PromptDataType = "text") -> ConverterResult:
        """
        Converts the given prompt into a URL-encoded string.
        """
        if not self.input_supported(input_type):
            raise ValueError("Input type not supported")

        return ConverterResult(output_text=urllib.parse.quote(prompt), output_type="text")
