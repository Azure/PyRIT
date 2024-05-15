# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import abc
import logging
from typing import Optional
from uuid import uuid4

from pyrit.memory import MemoryInterface
from pyrit.models import ChatMessage
from pyrit.orchestrator import Orchestrator
from pyrit.prompt_normalizer import PromptNormalizer
from pyrit.prompt_target import PromptTarget, PromptChatTarget
from pyrit.prompt_converter import PromptConverter

logger = logging.getLogger(__name__)


class CompletionState:
    def __init__(self, is_complete: bool):
        self.is_complete = is_complete


CRESCENDO_VARIANT_1 = 1


class CrescendoOrchestrator(Orchestrator):
    _memory: MemoryInterface

    def __init__(
        self,
        *,
        crescendo_variant: int,
        prompt_target: PromptTarget,
        red_teaming_chat: PromptChatTarget,
        prompt_converters: Optional[list[PromptConverter]] = None,
        memory: Optional[MemoryInterface] = None,
        memory_labels: dict[str, str] = None,
        verbose: bool = False,
    ) -> None:
        """Creates an orchestrator to manage conversations between a red teaming target and a prompt target.
           This is an abstract class and needs to be inherited.

        Args:
            crescendo_variant: 1-5 different variants supported; will load the variant from the corresponding YAML file.
            prompt_target: The target to send the prompts to.
            red_teaming_chat: The endpoint that creates prompts that are sent to the prompt target.
            prompt_converters: The prompt converters to use to convert the prompts before sending them to the prompt
                target. The converters are not applied on messages to the red teaming target.
            memory: The memory to use to store the chat messages. If not provided, a DuckDBMemory will be used.
            memory_labels: The labels to use for the memory. This is useful to identify the bot messages in the memory.
            verbose: Whether to print debug information.
        """

        super().__init__(
            prompt_converters=prompt_converters, memory=memory, memory_labels=memory_labels, verbose=verbose
        )

        self._prompt_target = prompt_target

        self._prompt_normalizer = PromptNormalizer(memory=self._memory)
        self._prompt_target._memory = self._memory
        self._prompt_target_conversation_id = str(uuid4())
        self._red_teaming_chat_conversation_id = str(uuid4())
        self._red_teaming_chat = red_teaming_chat
        self._red_teaming_chat._memory = self._memory
        self._crescendo_variant = crescendo_variant

    def get_memory(self):
        """
        Retrieves the memory associated with the red teaming orchestrator.
        """
        return self._memory.get_prompt_entries_by_orchestrator(self)

    @abc.abstractmethod
    def is_conversation_complete(self, messages: list[ChatMessage], *, red_teaming_chat_role: str) -> bool:
        """Returns True if the conversation is complete, False otherwise."""
        pass

    def apply_attack_strategy_until_completion(self, max_turns: int = 5):
        pass
