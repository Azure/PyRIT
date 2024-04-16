# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import codecs

from pyrit.models import PromptDataType
from pyrit.prompt_converter import PromptConverter


class ROT13Converter(PromptConverter):
    def convert(self, *, prompt: str, input_type: PromptDataType = "text") -> str:
        """
        Simple converter that just ROT13 encodes the prompts
        """
        if not self.is_supported(input_type):
            raise ValueError("Input type not supported")

        return codecs.encode(prompt, "rot13")

    def is_supported(self, input_type: PromptDataType) -> bool:
        return input_type == "text"
