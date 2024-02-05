# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from pyrit.prompt_transformer import PromptTransformer


class NoOpTransformer(PromptTransformer):
    def transform(self, prompt: str) -> str:
        """
        By default, the base transformer class does nothing to the prompt.
        """
        return prompt
