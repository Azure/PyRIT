# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import abc
import logging
from typing import Literal, Tuple

from pyrit.models import PromptDataType
from pyrit.prompt_converter import ConverterResult, PromptConverter

logger = logging.getLogger(__name__)


class SmugglerConverter(PromptConverter, abc.ABC):
    """
    Abstract base class for token smuggling converters.

    Provides the common asynchronous conversion interface and enforces
    implementation of encode_message and decode_message in subclasses.
    """

    def __init__(self, action: Literal["encode", "decode"] = "encode") -> None:
        if action not in ["encode", "decode"]:
            raise ValueError("Action must be either 'encode' or 'decode'")
        self.action = action

    async def convert_async(self, *, prompt: str, input_type: PromptDataType = "text") -> ConverterResult:
        """
        Convert the prompt by either encoding or decoding it based on the specified action.

        Args:
            prompt (str): The prompt to be processed.
            input_type (PromptDataType): Type of input; only "text" is supported.

        Returns:
            ConverterResult: The result containing the output text and its type.

        Raises:
            ValueError: If the input type is unsupported.
        """
        if not self.input_supported(input_type):
            raise ValueError("Input type not supported")
        if self.action == "encode":
            summary, encoded = self.encode_message(message=prompt)
            logger.info(f"Encoded message summary: {summary}")
            return ConverterResult(output_text=encoded, output_type="text")
        else:
            decoded = self.decode_message(message=prompt)
            return ConverterResult(output_text=decoded, output_type="text")

    def input_supported(self, input_type: PromptDataType) -> bool:
        """Return True if the input type is 'text'."""
        return input_type == "text"

    def output_supported(self, output_type: PromptDataType) -> bool:
        """Return True if the output type is 'text'."""
        return output_type == "text"

    @abc.abstractmethod
    def encode_message(self, *, message: str) -> Tuple[str, str]:
        """
        Encodes the given message.

        Must be implemented by subclasses.

        Args:
            message (str): The message to encode.

        Returns:
            Tuple[str, str]: A tuple containing a summary and the encoded message.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def decode_message(self, *, message: str) -> str:
        """
        Decodes the given message.

        Must be implemented by subclasses.

        Args:
            message (str): The encoded message.

        Returns:
            str: The decoded message.
        """
        raise NotImplementedError
