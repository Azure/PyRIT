# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import re

from pyrit.models import PromptDataType
from pyrit.prompt_converter import ConverterResult, PromptConverter


class CharacterSpaceConverter(PromptConverter):
    """
    Spaces out the input prompt and removes specified punctuations.

    For more information on the bypass strategy, refer to:
    https://www.robustintelligence.com/blog-posts/bypassing-metas-llama-classifier-a-simple-jailbreak
    """

    async def convert_async(self, *, prompt: str, input_type: PromptDataType = "text") -> ConverterResult:
        """Converts the given prompt by removing punctuation and spacing out characters."""
        if not self.input_supported(input_type):
            raise ValueError("Input type not supported")
        converted_text = re.sub("[!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~]", "", " ".join(prompt))
        return ConverterResult(output_text=converted_text, output_type="text")

    def input_supported(self, input_type: PromptDataType) -> bool:
        return input_type == "text"

    def output_supported(self, output_type: PromptDataType) -> bool:
        return output_type == "text"
