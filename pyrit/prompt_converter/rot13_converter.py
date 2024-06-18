# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import codecs

from pyrit.models import PromptDataType
from pyrit.prompt_converter import PromptConverter, ConverterResult


class ROT13Converter(PromptConverter):
    def convert(self, *, prompt: str, input_type: PromptDataType = "text") -> ConverterResult:
        """
        Simple converter that just ROT13 encodes the prompts
        """
        if not self.input_supported(input_type):
            raise ValueError("Input type not supported")

        return ConverterResult(output_text=codecs.encode(prompt, "rot13"), output_type="text")

    def input_supported(self, input_type: PromptDataType) -> bool:
        return input_type == "text"
