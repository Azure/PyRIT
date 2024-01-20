# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import abc

class PromptTransformer(abc.ABC):
    def transofrm(self, prompt) -> str:
        """
        By default, the base transformer class does nothing to the prompt.
        """
        return prompt