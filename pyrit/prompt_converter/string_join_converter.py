# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from pyrit.prompt_converter import PromptConverter


class StringJoinConverter(PromptConverter):

    def __init__(self, join_value="-"):
        self.join_value = join_value

    def convert(self, prompts: list[str], include_original: bool = False) -> list[str]:
        """
        Simple converter that uses str join for letters between. E.g. with a `-`
        it converts a prompt of `test` to `t-e-s-t`

        This can sometimes bypass LLM logic

        Args:
            prompt (str): The prompt to be converted.
            include_original (bool): Whether or not to include original prompt in the output

        Returns:
            list[str]: The converted prompt.
        """
        ret_list = prompts[:] if include_original else []

        for prompt in prompts:
            ret_list.append(self.join_value.join(prompt))

        return ret_list
