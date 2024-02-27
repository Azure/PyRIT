# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from pyrit.prompt_converter import PromptConverter


class NoOpConverter(PromptConverter):
    def convert(self, prompt: str) -> str:
        """
        By default, the base transformer class does nothing to the prompt.
        """
        return prompt
