# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import abc


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
