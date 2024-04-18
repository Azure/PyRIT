# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from pyrit.memory.memory_models import PromptMemoryEntry
from pyrit.models import PromptRequestResponse, PromptRequestPiece
from pyrit.prompt_target import PromptTarget
from pyrit.memory import MemoryInterface


class PromptChatTarget(PromptTarget):

    def __init__(self, *, memory: MemoryInterface) -> None:
        super().__init__(memory=memory)

    def set_system_prompt(
        self,
        *,
        system_prompt: str,
        conversation_id: str,
        orchestrator_identifier: dict[str, str],
        labels: dict,
    ) -> None:
        """
        Sets the system prompt for the prompt target. May be overridden by subclasses.
        """
        messages = self._memory.get_prompt_entries_with_conversation_id(conversation_id=conversation_id)

        if messages:
            raise RuntimeError("Conversation already exists, system prompt needs to be set at the beginning")

        system_entry = PromptMemoryEntry(
            entry=PromptRequestPiece(
                role="system",
                conversation_id=conversation_id,
                sequence=0,
                original_prompt_text=system_prompt,
                converted_prompt_text=system_prompt,
                prompt_target_identifier=self.get_identifier(),
                orchestrator_identifier=orchestrator_identifier,
                labels=labels,
            )
        )

        self._memory.insert_prompt_entries(entries=[system_entry])

    def send_chat_prompt(
        self,
        *,
        prompt: str,
        conversation_id: str,
        orchestrator_identifier: dict[str, str],
        labels: dict,
    ) -> PromptRequestResponse:
        """
        Sends a text prompt to the target without having to build the prompt request.
        """

        request = PromptRequestResponse(
            request_pieces=[
                PromptRequestPiece(
                    role="user",
                    conversation_id=conversation_id,
                    original_prompt_text=prompt,
                    converted_prompt_text=prompt,
                    prompt_target_identifier=self.get_identifier(),
                    orchestrator_identifier=orchestrator_identifier,
                    labels=labels,
                )
            ]
        )

        return self.send_prompt(prompt_request=request)

    async def send_chat_prompt_async(
        self,
        *,
        prompt: str,
        conversation_id: str,
        orchestrator_identifier: dict[str, str],
        labels: dict,
    ) -> PromptRequestResponse:
        """
        Sends a text prompt to the target without having to build the prompt request.
        """

        request = PromptRequestResponse(
            request_pieces=[
                PromptRequestPiece(
                    role="user",
                    conversation_id=conversation_id,
                    original_prompt_text=prompt,
                    converted_prompt_text=prompt,
                    prompt_target_identifier=self.get_identifier(),
                    orchestrator_identifier=orchestrator_identifier,
                    labels=labels,
                )
            ]
        )

        return await self.send_prompt_async(prompt_request=request)
