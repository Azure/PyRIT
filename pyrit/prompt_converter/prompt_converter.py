# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import abc
import json


class PromptConverter(abc.ABC):
    """
    A prompt converter is responsible for converting prompts into multiple representations.
    """

    @abc.abstractmethod
    def convert(self, prompts: list[str]) -> list[str]:
        """
        Converts the given prompts into multiple representations.

        Args:
            prompts (list[str]): The prompts to be converted.

        Returns:
            list[str]: The converted representations of the prompts.
        """
        pass

    @abc.abstractmethod
    def is_one_to_one_converter(self) -> bool:
        """Indicates if the conversion results in exactly one resulting prompt."""
        pass

    def to_dict(self):
        public_attributes = {k: v for k, v in self.__dict__.items() if not k.startswith("_")}
        public_attributes["__type__"] = self.__class__.__name__
        public_attributes["__module__"] = self.__class__.__module__
        return public_attributes


class PromptConverterList:
    def __init__(self, converters: list[PromptConverter]) -> None:
        self.converters = converters

    def to_json(self):
        return json.dumps([converter.to_dict() for converter in self.converters])
