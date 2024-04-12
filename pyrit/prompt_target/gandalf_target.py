# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging

from pyrit.completion import GandalfCompletionEngine, GandalfLevel
from pyrit.memory import DuckDBMemory, MemoryInterface
from pyrit.memory.memory_models import PromptRequestResponse
from pyrit.models.models import ChatMessage
from pyrit.prompt_target import PromptTarget


logger = logging.getLogger(__name__)


class GandalfTarget(GandalfCompletionEngine, PromptTarget):
    def __init__(
        self,
        *,
        level: GandalfLevel,
        memory: MemoryInterface = None,
    ) -> None:
        super().__init__(level=level)
        self._memory = memory if memory else DuckDBMemory()

    def send_prompt(       
        self,
        *,
        prompt_request: PromptRequestResponse
    ) -> PromptRequestResponse:
        
        prompt_request = prompt_request.request_pieces[0]

        messages = self._memory.get_chat_messages_with_conversation_id(
            conversation_id=prompt_request.conversation_id)

        prompt_request.sequence = len(messages)
        self._memory.insert_prompt_entries([prompt_request])

        logger.info(f"Sending the following prompt to the prompt target: {prompt_request}")

        response = super().complete_text(text=prompt_request.converted_prompt_text)

        if not response.completion:
            raise ValueError("The chat returned an empty response.")

        logger.info(f'Received the following response from the prompt target "{response.completion}"')


        response_entry = self._memory.add_response_entries_to_memory(
            request=prompt_request,
            response_text_pieces=[response.completion]
        )

        return response_entry


    async def send_prompt_async(
        self,
        *,
        prompt_request: PromptRequestResponse
    ) -> PromptRequestResponse:
        raise NotImplementedError()
