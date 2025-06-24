# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import base64

from pyrit.models import PromptDataType
from pyrit.prompt_converter import ConverterResult, PromptConverter


class Base64Converter(PromptConverter):
    """Converter that encodes text to base64 format.

    This converter takes input text and converts it to base64 encoding,
    which can be useful for obfuscating text or testing how systems
    handle encoded content.
    """

    async def convert_async(self, *, prompt: str, input_type: PromptDataType = "text") -> ConverterResult:
        """Convert the given prompt to base64 encoding.

        Args:
            prompt: The prompt to be converted.
            input_type: Type of data, unused for this converter.

        Returns:
            The converted text representation of the original prompt in base64.

        Raises:
            ValueError: If the input type is not supported.
        """
        if not self.input_supported(input_type):
            raise ValueError("Input type not supported")

        string_bytes = prompt.encode("utf-8")
        encoded_bytes = base64.b64encode(string_bytes)
        return ConverterResult(output_text=encoded_bytes.decode("utf-8"), output_type="text")

    def input_supported(self, input_type: PromptDataType) -> bool:
        return input_type == "text"

    def output_supported(self, output_type: PromptDataType) -> bool:
        return output_type == "text"
