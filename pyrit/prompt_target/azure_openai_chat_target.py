# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from pyrit.chat import AzureOpenAIChat
from pyrit.memory import FileMemory, MemoryInterface
from pyrit.models import ChatMessage
from pyrit.prompt_target import PromptChatTarget


class AzureOpenAIChatTarget(AzureOpenAIChat, PromptChatTarget):
    def __init__(
        self,
        *,
        deployment_name: str = None,
        endpoint: str = None,
        api_key: str = None,
        memory: MemoryInterface = None,
        api_version: str = "2023-08-01-preview",
        temperature: float = 1.0,
    ) -> None:
        AzureOpenAIChat.__init__(
            self, deployment_name=deployment_name, endpoint=endpoint, api_key=api_key, api_version=api_version
        )
        PromptChatTarget.__init__(self, memory=memory)

        self._memory = memory if memory else FileMemory()
        self._temperature = temperature

    def set_system_prompt(self, *, prompt: str, conversation_id: str, normalizer_id: str) -> None:
        messages = self._memory.get_memories_with_conversation_id(conversation_id=conversation_id)

        if messages:
            raise RuntimeError("Conversation already exists, system prompt needs to be set at the beginning")

        self._memory.add_chat_message_to_memory(
            conversation=ChatMessage(role="system", content=prompt),
            conversation_id=conversation_id,
            normalizer_id=normalizer_id,
        )

    def send_prompt(self, *, normalized_prompt: str, conversation_id: str, normalizer_id: str) -> str:
        messages = self._prepare_message(normalized_prompt, conversation_id, normalizer_id)

        resp = self.complete_chat(messages=messages, temperature=self._temperature)

        self._memory.add_chat_message_to_memory(
            ChatMessage(role="assistant", content=resp),
            conversation_id=conversation_id,
            normalizer_id=normalizer_id,
        )

        return resp

    async def send_prompt_async(self, normalized_prompt: str, conversation_id: str, normalizer_id: str) -> str:
        messages = self._prepare_message(normalized_prompt, conversation_id, normalizer_id)

        resp = await super().complete_chat_async(messages=messages, temperature=self.temperature)

        self._memory.add_chat_message_to_memory(
            ChatMessage(role="assistant", content=resp),
            conversation_id=conversation_id,
            normalizer_id=normalizer_id,
        )

        return resp

    def _prepare_message(self, normalized_prompt: str, conversation_id: str, normalizer_id: str):
        messages = self._memory.get_chat_messages_with_conversation_id(conversation_id=conversation_id)
        msg = ChatMessage(role="user", content=normalized_prompt)
        messages.append(msg)
        self._memory.add_chat_message_to_memory(msg, conversation_id, normalizer_id)
        return messages
