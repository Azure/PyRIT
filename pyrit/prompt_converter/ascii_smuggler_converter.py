# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Literal
from pyrit.models import PromptDataType
from pyrit.prompt_converter import PromptConverter, ConverterResult
import logging

logger = logging.getLogger(__name__)


class AsciiSmugglerConverter(PromptConverter):
    """
    Converts a string to or from ASCII smuggling encoded format.

    Replicates the functionality detailed in the following blog post:
    https://embracethered.com/blog/posts/2024/hiding-and-finding-text-with-unicode-tags/

    """

    def __init__(self, action: Literal["encode", "decode"] = "encode", unicode_tags: bool = False):
        """
        Initialize the converter with options for encoding/decoding.

        Args:
            unicode_tags (bool): Whether to add Unicode tags during encoding.
        """
        self.unicode_tags = unicode_tags
        self.action = action

        if self.action not in ["encode", "decode"]:
            raise ValueError("Action must be either 'encode' or 'decode'")

    async def convert_async(self, *, prompt: str, input_type: PromptDataType = "text") -> ConverterResult:
        """
        Process the prompt to either encode or decode based on options and prompt type.
        Args:
            prompt (str): The prompt to be converted.
            input_type (PromptDataType): The type of input, defaults to "text".
        Returns:
            ConverterResult: The converted result.
        """
        if not self.input_supported(input_type):
            raise ValueError("Input type not supported")

        # Determine action based on input: either "encode" or "decode"

        if self.action == "encode":
            code_points, encoded = self.encode_message(message=prompt)
            logger.info(f"Encoded message: {code_points}")
            return ConverterResult(output_text=encoded, output_type="text")

        else:  # decode
            # Look into and document the "unicode_escape" decoding
            # input_text = prompt.encode("ascii", "ignore").decode("unicode_escape")
            decoded_message = self.decode_message(message=prompt)
            return ConverterResult(output_text=decoded_message, output_type="text")

    def input_supported(self, input_type: PromptDataType) -> bool:
        return input_type == "text"

    def encode_message(self, *, message: str):
        """Encode the message using Unicode tags"""
        encoded = ""
        code_points = ""
        invalid_chars = ""
        if self.unicode_tags:
            encoded += chr(0xE0001)
            code_points += "U+E0001 "
        for char in message:
            if 0x20 <= ord(char) <= 0x7E:
                code_point = 0xE0000 + ord(char)
                encoded += chr(code_point)
                code_points += f"U+{code_point:X} "
            else:
                invalid_chars += char
        if self.unicode_tags:
            encoded += chr(0xE007F)
            code_points += "U+E007F"

        if invalid_chars:
            logger.error(f"Invalid characters detected: {invalid_chars}")
        return code_points, encoded

    def decode_message(self, *, message: str):
        """Decode the message from Unicode tags"""
        decoded_message = ""
        for char in message:
            code_point = ord(char)
            if 0xE0000 <= code_point <= 0xE007F:
                decoded_char = chr(code_point - 0xE0000)
                if not 0x20 <= ord(decoded_char) <= 0x7E:
                    logger.info(f"Potential unicode tag detected: {decoded_char}")
                    pass
                else:
                    decoded_message += decoded_char

        if len(decoded_message) != len(message):
            logger.info("Hidden Unicode Tags discovered.")
        else:
            logger.info("No hidden Unicode Tag characters discovered.")
        return decoded_message
