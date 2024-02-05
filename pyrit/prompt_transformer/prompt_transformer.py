# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import abc


class PromptTransformer(abc.ABC):
    @abc.abstractmethod
    def transform(self, prompt: str) -> str:
        """
        By default, the base transformer class does nothing to the prompt.
        """
        pass
