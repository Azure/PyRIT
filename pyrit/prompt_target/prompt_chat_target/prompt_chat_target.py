# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import abc

from pyrit.memory.memory_models import PromptRequestResponse
from pyrit.prompt_target import PromptTarget
from pyrit.memory import MemoryInterface


class PromptChatTarget(PromptTarget):

    def __init__(self, *, memory: MemoryInterface) -> None:
        super().__init__(memory=memory)

    @abc.abstractmethod
    def set_system_prompt(
        self,
        *,
        prompt_request: PromptRequestResponse
    ) -> None:
        """
        Sets the system prompt for the prompt target
        """
