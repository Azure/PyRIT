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
        self._memory.add_request_pieces_to_memory(request_pieces=prompt_request.request_pieces)

        return None

    async def send_prompt_async(self, *, prompt_request: PromptRequestResponse) -> PromptRequestResponse:

        await asyncio.sleep(0)

        return self.send_prompt(prompt_request=prompt_request)
