# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import abc
from dataclasses import dataclass

from pyrit.models import PromptDataType
from pyrit.models.identifiers import Identifier


@dataclass
class ConverterResult:
    output_text: str
    output_type: PromptDataType

    def __str__(self):
        return f"{self.output_type}: {self.output_text}"


class PromptConverter(abc.ABC, Identifier):
    """
    A prompt converter is responsible for converting prompts into a different representation.

    """

    @abc.abstractmethod
    def convert(self, *, prompt: str, input_type: PromptDataType = "text") -> ConverterResult:
        """
        Converts the given prompts into a different representation

        Args:
            prompt: The prompt to be converted.

        Returns:
            str: The converted representation of the prompts.
        """
        pass

    @abc.abstractmethod
    def input_supported(self, input_type: PromptDataType) -> bool:
        """
        Checks if the input type is supported by the converter

        Args:
            input_type: The input type to check

        Returns:
            bool: True if the input type is supported, False otherwise
        """
        pass

    def get_identifier(self):
        public_attributes = {}
        public_attributes["__type__"] = self.__class__.__name__
        public_attributes["__module__"] = self.__class__.__module__
        return public_attributes
