# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import concurrent.futures
import asyncio

from pyrit.models import PromptDataType
from pyrit.prompt_converter import PromptConverter, ConverterResult


class UnicodeSubstitutionConverter(PromptConverter):
    def __init__(self, *, start_value=0xE0000):
        self.startValue = start_value

    def convert(self, *, prompt: str, input_type: PromptDataType = "text") -> ConverterResult:
        """
        Deprecated. Use async_convert instead.
        """
        pool = concurrent.futures.ThreadPoolExecutor()
        return pool.submit(asyncio.run, self.async_convert(prompt=prompt, input_type=input_type)).result()

    async def async_convert(self, *, prompt: str, input_type: PromptDataType = "text") -> ConverterResult:
        """
        Simple converter that just encodes the prompt using any unicode starting point.
        Default is to use invisible flag emoji characters.
        """
        if not self.input_supported(input_type):
            raise ValueError("Input type not supported")

        ret_text = "".join(chr(self.startValue + ord(ch)) for ch in prompt)

        return ConverterResult(output_text=ret_text, output_type="text")

    def input_supported(self, input_type: PromptDataType) -> bool:
        return input_type == "text"
