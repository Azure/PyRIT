# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import asyncio
from io import TextIOBase
import sys

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

    def __init__(self, *, io: TextIOBase = sys.stdout, memory: MemoryInterface = None) -> None:
        super().__init__(memory)
        self.io = io

    def send_prompt(self, normalized_prompt: str, conversation_id: str, normalizer_id: str) -> str:
        msg = ChatMessage(role="user", content=normalized_prompt)
        self.io.write(msg)

        self.memory.add_chat_message_to_memory(
            conversation=msg, conversation_id=conversation_id, normalizer_id=normalizer_id
        )

        return ""

    async def send_prompt_async(self, normalized_prompt: str, conversation_id: str, normalizer_id: str) -> str:
        self.send_prompt(normalized_prompt, conversation_id, normalizer_id)
        await asyncio.sleep(0)
        return ""