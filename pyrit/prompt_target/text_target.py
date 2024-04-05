# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import asyncio
import sys

from typing import IO

from pyrit.memory import MemoryInterface
from pyrit.memory.memory_models import PromptRequestResponse
from pyrit.models import ChatMessage
from pyrit.prompt_normalizer.prompt_request_piece import PromptRequestPieces
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

    def send_prompt(
        self,
        *,
        prompt_request: PromptRequestResponse,
        verbose: bool = False
    ) -> PromptRequestPieces:
        
        self._text_stream.write(f"{str(prompt_request)}\n")
        self._memory.insert_prompt_entries(entries=prompt_request.request_pieces)

        return None
    

    async def send_prompt_async(
        self,
        *,
        prompt_request: PromptRequestResponse,
        verbose: bool = False
    ) -> PromptRequestPieces:
        
        await asyncio.sleep(0)

        return self.send_prompt(
            prompt_request=prompt_request,
            verbose=verbose
        )
