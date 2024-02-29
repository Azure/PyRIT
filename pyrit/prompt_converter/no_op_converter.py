# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from pyrit.prompt_converter import PromptConverter


class NoOpConverter(PromptConverter):
    def convert(self, prompts: list[str], include_original: bool = False) -> list[str]:
        """
        By default, the base converter class does nothing to the prompt.
        """
        return prompts
