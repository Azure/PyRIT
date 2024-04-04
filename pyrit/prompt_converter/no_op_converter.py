# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from pyrit.memory.memory_models import PromptDataType
from pyrit.prompt_converter import PromptConverter

class NoOpConverter(PromptConverter):
    def convert(self, prompt: str, input_type: PromptDataType) -> str:
        """
        By default, the base converter class does nothing to the prompt.
        """
        return prompt

    def is_supported(self, input_type: PromptDataType) -> bool:
        return True