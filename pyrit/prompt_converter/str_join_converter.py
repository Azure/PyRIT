# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from pyrit.prompt_converter import PromptConverter


class StrJoinConverter(PromptConverter):

    def __init__(self, join_value="-"):
        self.join_value = join_value

    def convert(self, prompt: str) -> str:
        """
        Simple converter that uses str join for letters between

        This can sometimes bypass LLM logic
        """
        return self.join_value.join(prompt)
