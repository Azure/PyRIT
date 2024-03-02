# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import codecs

from pyrit.prompt_converter import PromptConverter


class ROT13Converter(PromptConverter):
    def convert(self, prompt: str) -> list[str]::
        """
        Simple converter that just ROT13 encodes the prompt
        """
        encoded_bytes = codecs.encode(prompt, "rot13")
        return encoded_bytes
