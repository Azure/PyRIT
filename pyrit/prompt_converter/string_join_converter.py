# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from pyrit.prompt_converter import PromptConverter


class StringJoinConverter(PromptConverter):

    def __init__(self, *, join_value="-", include_original=False):
        self.join_value = join_value
        self.include_original = include_original

    def convert(self, prompts: list[str]) -> list[str]:
        """
        Simple converter that uses str join for letters between. E.g. with a `-`
        it converts a prompt of `test` to `t-e-s-t`

        This can sometimes bypass LLM logic

        Args:
            prompt (str): The prompt to be converted.
            include_original (bool): Whether or not to include original prompt in the output

        Returns:
            list[str]: The converted prompts.
        """
        return [self.join_value.join(prompt) for prompt in prompts]
