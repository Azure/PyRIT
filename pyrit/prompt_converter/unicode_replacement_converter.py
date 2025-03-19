# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from pyrit.models import PromptDataType
from pyrit.prompt_converter import ConverterResult, PromptConverter


class UnicodeReplacementConverter(PromptConverter):

    def __init__(self, encode_spaces: bool = False):
        """
        Initializes a UnicodeReplacementConverter object.

        Args:
            encode_spaces (bool): If True, spaces in the prompt will be replaced with unicode representation.
                                  Default is False.
        """
        self.encode_spaces = encode_spaces

    async def convert_async(self, *, prompt: str, input_type: PromptDataType = "text") -> ConverterResult:
        """
        Simple converter that returns the unicode representation of the prompt.
        """
        if not self.input_supported(input_type):
            raise ValueError("Input type not supported")

        ret_text = "".join(f"\\u{ord(ch):04x}" for ch in prompt)
        if not self.encode_spaces:
            ret_text = ret_text.replace("\\u0020", " ")

        return ConverterResult(output_text=ret_text, output_type="text")

    def input_supported(self, input_type: PromptDataType) -> bool:
        return input_type == "text"

    def output_supported(self, output_type: PromptDataType) -> bool:
        return output_type == "text"
