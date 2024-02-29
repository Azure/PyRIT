# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import base64

from pyrit.prompt_converter import PromptConverter


class Base64Converter(PromptConverter):
    def __init__(self, include_original=False):
        self.include_original = include_original

    def convert(self, prompts: list[str]) -> list[str]:
        """
        Simple converter that just base64 encodes the prompt
        """
        ret_list = prompts[:] if self.include_original else []

        for prompt in prompts:
            string_bytes = prompt.encode("utf-8")
            encoded_bytes = base64.b64encode(string_bytes)
            ret_list.append(encoded_bytes.decode("utf-8"))

        return ret_list
