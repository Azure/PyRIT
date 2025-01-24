# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import abc
from typing import Optional

from pyrit.models import PromptRequestPiece
from pyrit.prompt_target import PromptTarget


class PromptChatTarget(PromptTarget):
    """
    A prompt chat target is a target where you can explicitly set the conversation history using memory.

    Some algorithms require conversation to be modified (e.g. deleting the last message) or set explicitly.
    These algorithms will require PromptChatTargets be used.

    As a concrete example, OpenAI chat targets are PromptChatTargets. You can set made-up conversation history.
    Realtime chat targets or OpenAI completions are NOT PromptChatTargets. You don't send the conversation history.
    """

    def __init__(self, *, max_requests_per_minute: Optional[int] = None) -> None:
        super().__init__(max_requests_per_minute=max_requests_per_minute)

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

    @abc.abstractmethod
    def is_json_response_supported(self) -> bool:
        """
        Abstract method to determine if JSON response format is supported by the target.

        Returns:
            bool: True if JSON response is supported, False otherwise.
        """
        pass

    def is_response_format_json(self, request_piece: PromptRequestPiece) -> bool:
        """
        Checks if the response format is JSON and ensures the target supports it.

        Args:
            request_piece: A PromptRequestPiece object with a `prompt_metadata` dictionary that may
                include a "response_format" key.

        Returns:
            bool: True if the response format is JSON and supported, False otherwise.

        Raises:
            ValueError: If "json" response format is requested but unsupported.
        """
        if request_piece.prompt_metadata:
            response_format = request_piece.prompt_metadata.get("response_format")
            if response_format == "json":
                if not self.is_json_response_supported():
                    target_name = self.get_identifier()["__type__"]
                    raise ValueError(f"This target {target_name} does not support JSON response format.")
                return True
        return False
