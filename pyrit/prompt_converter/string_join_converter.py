# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from pyrit.models import PromptDataType
from pyrit.prompt_converter import PromptConverter


class StringJoinConverter(PromptConverter):

    def __init__(self, *, join_value="-"):
        self.join_value = join_value

    def convert(self, *, prompt: str, input_type: PromptDataType = "text") -> str:
        """
        Simple converter that uses str join for letters between. E.g. with a `-`
        it converts a prompt of `test` to `t-e-s-t`

        This can sometimes bypass LLM logic

        Args:
            prompt (str): The prompt to be converted.

        Returns:
            list[str]: The converted prompts.
        """
        if not self.is_supported(input_type):
            raise ValueError("Input type not supported")

        return self.join_value.join(prompt)

    def is_supported(self, input_type: PromptDataType) -> bool:
        return input_type == "text"
