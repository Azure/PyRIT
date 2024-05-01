# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import asyncio
import sys

from typing import IO

from pyrit.memory import MemoryInterface
from pyrit.models import PromptRequestResponse
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

    def send_prompt(self, *, prompt_request: PromptRequestResponse) -> PromptRequestResponse:

        self._text_stream.write(f"{str(prompt_request)}\n")
        self._memory.add_request_response_to_memory(request=prompt_request)

        return None

    async def send_prompt_async(self, *, prompt_request: PromptRequestResponse) -> PromptRequestResponse:

        self._validate_request(prompt_request=prompt_request)
        await asyncio.sleep(0)

        return self.send_prompt(prompt_request=prompt_request)

    def _validate_request(self, *, prompt_request: PromptRequestResponse) -> None:
        if len(prompt_request.request_pieces) != 1:
            raise ValueError("This target only supports a single prompt request piece.")

        request = prompt_request.request_pieces[0]
        messages = self._memory.get_chat_messages_with_conversation_id(conversation_id=request.conversation_id)

        if len(messages) > 0:
            raise ValueError("This target only supports a single turn conversation.")
