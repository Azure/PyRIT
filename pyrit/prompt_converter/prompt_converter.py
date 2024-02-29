# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import abc


class PromptConverter(abc.ABC):
    @abc.abstractmethod
    def convert(self, prompts: list[str], include_original: bool = False) -> list[str]:
        """
        This is the class that changes a prompt into N different representations (N>0).

        In this sense, a prompt is any input we give to an LLM endpoint. It can be text, code, an image, etc.

        Currently, there are base64, Unicode, but also there will be "word doc" and "pdf" converters.
        These converters can also be probabilistic, like using an LLM to find N prompt variations.
        """
        pass
