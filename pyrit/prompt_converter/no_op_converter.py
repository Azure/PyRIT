# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from pyrit.prompt_converter import PromptConverter


class NoOpConverter(PromptConverter):
    def convert(self, prompts: list[str]) -> list[str]:
        """
        By default, the base converter class does nothing to the prompt.
        """
        return prompts

    def is_one_to_one_converter(self) -> bool:
        return True
