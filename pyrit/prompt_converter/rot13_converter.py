# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import codecs

from pyrit.prompt_converter import PromptConverter


class ROT13Converter(PromptConverter):
    def convert(self, prompts: list[str]) -> list[str]:
        """
        Simple converter that just ROT13 encodes the prompts
        """
        return [codecs.encode(prompt, "rot13") for prompt in prompts]

    def is_one_to_one_converter(self) -> bool:
        return True
