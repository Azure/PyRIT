# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from pyrit.chat import AMLOnlineEndpointChat
from pyrit.chat_message_normalizer import ChatMessageNop, ChatMessageNormalizer
from pyrit.memory import FileMemory, MemoryInterface
from pyrit.models import ChatMessage
from pyrit.prompt_target import PromptTarget


class AMLOnlineEndpointChatTarget(AMLOnlineEndpointChat, PromptTarget):
    def __init__(
        self,
        *,
        endpoint_uri: str = None,
        api_key: str = None,
        chat_message_normalizer: ChatMessageNormalizer = ChatMessageNop(),
        memory: MemoryInterface = None,
    ) -> None:
        super().__init__(endpoint_uri=endpoint_uri, api_key=api_key, chat_message_normalizer=chat_message_normalizer)

        self._memory = memory if memory else FileMemory()

    def set_system_prompt(self, prompt: str, conversation_id: str, normalizer_id: str) -> None:
        messages = self._memory.get_memories_with_conversation_id(conversation_id=conversation_id)

        if messages:
            raise RuntimeError("Conversation already exists, system prompt needs to be set at the beginning")

        self._memory.add_chat_message_to_memory(
            conversation=ChatMessage(role="system", content=prompt),
            conversation_id=conversation_id,
            normalizer_id=normalizer_id,
        )

    def send_prompt(self, normalized_prompt: str, conversation_id: str, normalizer_id: str) -> str:
        messages = self._memory.get_chat_messages_with_conversation_id(conversation_id=conversation_id)

        msg = ChatMessage(role="user", content=normalized_prompt)

        messages.append(msg)

        self._memory.add_chat_message_to_memory(
            conversation=msg, conversation_id=conversation_id, normalizer_id=normalizer_id
        )

        resp = super().complete_chat(messages=messages, temperature=self._temperature)

        self._memory.add_chat_message_to_memory(
            conversation=ChatMessage(role="assistant", content=resp),
            conversation_id=conversation_id,
            normalizer_id=normalizer_id,
        )

        return resp
