# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import concurrent.futures
import asyncio

from pyrit.models import PromptDataType
from pyrit.prompt_converter import PromptConverter, ConverterResult


class StringJoinConverter(PromptConverter):

    def __init__(self, *, join_value="-"):
        self.join_value = join_value

    def convert(self, *, prompt: str, input_type: PromptDataType = "text") -> ConverterResult:
        """
        Deprecated. Use async_convert instead.
        """
        pool = concurrent.futures.ThreadPoolExecutor()
        return pool.submit(asyncio.run, self.async_convert(prompt=prompt, input_type=input_type)).result()

    async def async_convert(self, *, prompt: str, input_type: PromptDataType = "text") -> ConverterResult:
        """
        Simple converter that uses str join for letters between. E.g. with a `-`
        it converts a prompt of `test` to `t-e-s-t`

        This can sometimes bypass LLM logic

        Args:
            prompt (str): The prompt to be converted.

        Returns:
            list[str]: The converted prompts.
        """
        if not self.input_supported(input_type):
            raise ValueError("Input type not supported")

        return ConverterResult(output_text=self.join_value.join(prompt), output_type="text")

    def input_supported(self, input_type: PromptDataType) -> bool:
        return input_type == "text"
