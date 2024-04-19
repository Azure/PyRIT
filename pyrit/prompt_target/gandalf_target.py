# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import asyncio
import concurrent.futures
import logging
from multiprocessing import pool

from pyrit.completion import GandalfLevel
from pyrit.common import net_utility
from pyrit.memory import DuckDBMemory, MemoryInterface
from pyrit.models import PromptRequestResponse
from pyrit.prompt_target import PromptTarget


logger = logging.getLogger(__name__)


class GandalfTarget(PromptTarget):
    def __init__(
        self,
        *,
        level: GandalfLevel,
        memory: MemoryInterface = None,
    ) -> None:
        self._memory = memory if memory else DuckDBMemory()

        self._endpoint = "https://gandalf.lakera.ai/api/send-message"
        self._defender = level.value

    def send_prompt(self, *, prompt_request: PromptRequestResponse) -> PromptRequestResponse:
        """
        Deprecated. Use send_prompt_async instead.
        """
        pool = concurrent.futures.ThreadPoolExecutor()
        return pool.submit(asyncio.run, self.send_prompt_async(prompt_request=prompt_request)).result()


    async def send_prompt_async(self, *, prompt_request: PromptRequestResponse) -> PromptRequestResponse:
        request = prompt_request.request_pieces[0]

        messages = self._memory.get_chat_messages_with_conversation_id(conversation_id=request.conversation_id)

        request.sequence = len(messages)
        self._memory.add_request_pieces_to_memory(request_pieces=[request])

        logger.info(f"Sending the following prompt to the prompt target: {request}")

        response = await self._complete_text_async(request.converted_prompt_text)

        response_entry = self._memory.add_response_entries_to_memory(
            request=request, response_text_pieces=[response]
        )

        return response_entry

    async def _complete_text_async(self, text: str) -> str:
        payload = {
            "defender": self._defender,
            "prompt": text,
        }

        resp = await net_utility.make_request_and_raise_if_error_async(
            endpoint_uri=self._endpoint, method="POST", request_body=payload, post_type="data"
        )

        if not resp.text:
            raise ValueError("The chat returned an empty response.")

        logger.info(f'Received the following response from the prompt target "{resp.text}"')
        return resp.text
