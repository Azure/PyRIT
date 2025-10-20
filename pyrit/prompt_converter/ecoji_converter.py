# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import io
from typing import Optional

import ecoji

from pyrit.models import PromptDataType
from pyrit.prompt_converter import PromptConverter, ConverterResult


class EcojiConverter(PromptConverter):
    """
    Converter that encodes text using Ecoji encoding.

    Ecoji is an encoding scheme that represents binary data using emojis.
    See https://ecoji.io/ for more details.
    """

    def __init__(self) -> None:
        """Initialize the Ecoji converter."""
        pass

    async def convert_async(self, *, prompt: str, input_type: PromptDataType = "text") -> ConverterResult:
        """
        Convert the given prompt to Ecoji encoding.

        Args:
            prompt (str): The text to encode.
            input_type (PromptDataType): The type of input. Defaults to "text".

        Returns:
            ConverterResult: The result containing the Ecoji-encoded text.

        Raises:
            ValueError: If the input type is not supported.
        """
        if not self.input_supported(input_type=input_type):
            raise ValueError("Input type not supported")

        encoded_text = self._encode_to_ecoji(prompt)
        return ConverterResult(output_text=encoded_text, output_type="text")

    def _encode_to_ecoji(self, text: str) -> str:
        """
        Encode text to Ecoji format.

        Args:
            text (str): The text to encode.

        Returns:
            str: The Ecoji-encoded text.
        """
        text_bytes = text.encode("utf-8")
        reader = io.BytesIO(text_bytes)
        writer = io.StringIO()

        ecoji.encode(reader, writer)

        return writer.getvalue()

    def input_supported(self, input_type: PromptDataType) -> bool:
        return input_type == "text"

    def output_supported(self, output_type: PromptDataType) -> bool:
        return output_type == "text"
