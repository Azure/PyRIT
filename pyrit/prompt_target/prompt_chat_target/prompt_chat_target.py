# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import abc

from pyrit.prompt_target import PromptTarget
from pyrit.memory import MemoryInterface
from pyrit.models import ChatMessage


class PromptChatTarget(PromptTarget):

    def __init__(self, *, memory: MemoryInterface) -> None:
        super().__init__(memory=memory)

    @abc.abstractmethod
    def set_system_prompt(self, *, prompt: str, conversation_id: str, normalizer_id: str) -> None:
        """
        Sets the system prompt for the prompt target
        """

    @abc.abstractmethod
    def complete_chat(self, *, messages: list[ChatMessage]) -> str:
        """Complete chat based on a message
           Update _chat_messages with the messages used in the completion.

        Args:
            messages: The list of chat messages to be used in the chat completion

        Returns:
            The completed chat
        """
