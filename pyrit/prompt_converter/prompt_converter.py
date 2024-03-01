# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import abc


class PromptConverter(abc.ABC):
    """
    A prompt converter is responsible for converting prompts into multiple representations.

    Attributes:
        include_original (bool): Flag indicating whether to include the original prompt in the
                                 converted representations.
    """

    include_original: bool = False

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
