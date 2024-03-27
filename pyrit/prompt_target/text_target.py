# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import asyncio
import sys

from typing import IO

from pyrit.memory import MemoryInterface
from pyrit.models import ChatMessage
from pyrit.prompt_target import PromptTarget


class TextTarget(PromptTarget):
    """
    The TextTarget takes prompts, adds them to memory and writes them to io
    which is sys.stdout by default

    This can be useful in various situations, for example, if operators want to generate prompts
    but enter them manually.
    """

    def __init__(self, *, text_stream: IO[str] = sys.stdout, memory: MemoryInterface = None) -> None:
        super().__init__(memory=memory)
        self.stream_name = text_stream.name
        self._text_stream = text_stream

    def send_prompt(self, *, normalized_prompt: str, conversation_id: str, normalizer_id: str) -> str:
        msg = ChatMessage(role="user", content=normalized_prompt)
        self._text_stream.write(f"{str(msg)}\n")

        self._memory.add_chat_message_to_memory(
            conversation=msg, conversation_id=conversation_id, normalizer_id=normalizer_id
        )

        return str(msg)

    async def send_prompt_async(self, *, normalized_prompt: str, conversation_id: str, normalizer_id: str) -> str:
        await asyncio.sleep(0)

        return self.send_prompt(
            normalized_prompt=normalized_prompt, conversation_id=conversation_id, normalizer_id=normalizer_id
        )
