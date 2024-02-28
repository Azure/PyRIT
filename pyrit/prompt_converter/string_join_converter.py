# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from pyrit.prompt_converter import PromptConverter


class StringJoinConverter(PromptConverter):

    def __init__(self, join_value="-"):
        self.join_value = join_value

    def convert(self, prompt: str) -> str:
        """
        Simple converter that uses str join for letters between. E.g. with a "-"
        it converts a promtp of test to t-e-s-t

        This can sometimes bypass LLM logic

        Args:
            prompt (str): The prompt to be converted.

        Returns:
            str: The converted prompt.
        """
        return self.join_value.join(prompt)
