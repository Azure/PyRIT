# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Any, Coroutine
from pyrit.completion import GandalfCompletionEngine, GandalfLevel
from pyrit.memory import DuckDBMemory, MemoryInterface
from pyrit.models import ChatMessage
from pyrit.prompt_target import PromptTarget


class GandalfTarget(GandalfCompletionEngine, PromptTarget):
    def __init__(
        self,
        *,
        level: GandalfLevel,
        memory: MemoryInterface = None,
    ) -> None:
        super().__init__(level=level)
        self._memory = memory if memory else DuckDBMemory()

    def set_system_prompt(self, *, prompt: str, conversation_id: str, normalizer_id: str) -> None:
        raise NotImplementedError("Cannot set system prompt with Gandalf.")

    def send_prompt(self, *, normalized_prompt: str, conversation_id: str, normalizer_id: str) -> str:
        msg = ChatMessage(role="user", content=normalized_prompt)

        self._memory.add_chat_message_to_memory(
            conversation=msg, conversation_id=conversation_id, normalizer_id=normalizer_id
        )

        response = super().complete_text(text=normalized_prompt)

        self._memory.add_chat_message_to_memory(
            conversation=ChatMessage(role="assistant", content=response.completion),
            conversation_id=conversation_id,
            normalizer_id=normalizer_id,
        )

        return response.completion

    def send_prompt_async(
        self, *, normalized_prompt: str, conversation_id: str, normalizer_id: str
    ) -> Coroutine[Any, Any, str]:
        return super().send_prompt_async(
            normalized_prompt=normalized_prompt, conversation_id=conversation_id, normalizer_id=normalizer_id
        )
