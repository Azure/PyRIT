# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from pyrit.memory import MemoryInterface
from pyrit.models import ChatMessage
from pyrit.prompt_target import PromptTarget


class NoOpTarget(PromptTarget):
    """
    The NoOpTarget takes prompts, adds them to memory and prints them, but doesn't send them anywhere

    This can be useful in various situations, for example, if operators want to generate prompts
    but enter them manually.
    """

    def __init__(self, *, memory: MemoryInterface = None) -> None:
        super().__init__(memory)

    def send_prompt(self, normalized_prompt: str, conversation_id: str, normalizer_id: str) -> str:
        msg = ChatMessage(role="user", content=normalized_prompt)
        print(msg)

        self._memory.add_chat_message_to_memory(
            conversation=msg, conversation_id=conversation_id, normalizer_id=normalizer_id
        )

        return ""
