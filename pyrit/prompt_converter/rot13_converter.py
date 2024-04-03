# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import codecs

from pyrit.memory.memory_models import PromptDataType
from pyrit.prompt_converter import PromptConverter


class ROT13Converter(PromptConverter):
    def convert(self, prompt: str) -> str:
        """
        Simple converter that just ROT13 encodes the prompts
        """
        return codecs.encode(prompt, "rot13")

    def is_supported(self, input_type: PromptDataType) -> bool:
        return input_type == "text"