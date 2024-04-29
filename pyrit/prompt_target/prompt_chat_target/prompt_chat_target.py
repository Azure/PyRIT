# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Optional
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
        orchestrator_identifier: Optional[dict[str, str]] = None,
        labels: Optional[dict[str, str]] = None,
    ) -> None:
        """
        Sets the system prompt for the prompt target. May be overridden by subclasses.
        """
        messages = self._memory.get_conversation(conversation_id=conversation_id)

        if messages:
            raise RuntimeError("Conversation already exists, system prompt needs to be set at the beginning")

        self._memory.add_request_response_to_memory(
            request=PromptRequestPiece(
                role="system",
                conversation_id=conversation_id,
                original_value=system_prompt,
                converted_value=system_prompt,
                prompt_target_identifier=self.get_identifier(),
                orchestrator_identifier=orchestrator_identifier,
                labels=labels,
            ).to_prompt_request_response()
        )

    def send_chat_prompt(
        self,
        *,
        prompt: str,
        conversation_id: str,
        orchestrator_identifier: Optional[dict[str, str]] = None,
        labels: Optional[dict[str, str]] = None,
    ) -> PromptRequestResponse:
        """
        Sends a text prompt to the target without having to build the prompt request.
        """

        request = PromptRequestResponse(
            request_pieces=[
                PromptRequestPiece(
                    role="user",
                    conversation_id=conversation_id,
                    original_value=prompt,
                    converted_value=prompt,
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
        orchestrator_identifier: Optional[dict[str, str]] = None,
        labels: Optional[dict[str, str]] = None,
    ) -> PromptRequestResponse:
        """
        Sends a text prompt to the target without having to build the prompt request.
        """

        request = PromptRequestResponse(
            request_pieces=[
                PromptRequestPiece(
                    role="user",
                    conversation_id=conversation_id,
                    original_value=prompt,
                    converted_value=prompt,
                    prompt_target_identifier=self.get_identifier(),
                    orchestrator_identifier=orchestrator_identifier,
                    labels=labels,
                )
            ]
        )

        return await self.send_prompt_async(prompt_request=request)
