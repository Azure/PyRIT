# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from pyrit.chat.azure_openai_chat import AzureOpenAIChat
from pyrit.memory import FileMemory, MemoryInterface
from pyrit.models import ChatMessage
from pyrit.prompt_target import PromptTarget


class AzureOpenAIChatTarget(PromptTarget):
    def __init__(self, *, memory: MemoryInterface = None) -> None:
        self.memory = memory if memory else FileMemory()

    def set_system_prompt(self, prompt: str, conversation_id: str, normalizer_id: str) -> None:
        messages = self.memory.get_memories_with_conversation_id(conversation_id=conversation_id)

        if messages:
            raise RuntimeError("Conversation already exists, system prompt needs to be set at the beginning")

        self.memory.add_chat_message_to_memory(
            conversation=ChatMessage(role="system", content=prompt),
            conversation_id=conversation_id,
            normalizer_id=normalizer_id,
        )

    def send_prompt(self, normalized_prompt: str, conversation_id: str, normalizer_id: str) -> str:
        messages = self.memory.get_chat_messages_with_conversation_id(conversation_id=conversation_id)

        msg = ChatMessage(role="user", content=normalized_prompt)

        messages.append(msg)

        self.memory.add_chat_message_to_memory(
            conversation=msg, conversation_id=conversation_id, normalizer_id=normalizer_id
        )

        resp = super().complete_chat(messages=messages, temperature=self.temperature)

        self.memory.add_chat_message_to_memory(
            conversation=ChatMessage(role="assistant", content=resp),
            conversation_id=conversation_id,
            normalizer_id=normalizer_id,
        )

        return resp
