# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import abc
from typing import Optional

from pyrit.models import MessagePiece
from pyrit.prompt_target import PromptTarget


class PromptChatTarget(PromptTarget):
    """
    A prompt chat target is a target where you can explicitly set the conversation history using memory.

    Some algorithms require conversation to be modified (e.g. deleting the last message) or set explicitly.
    These algorithms will require PromptChatTargets be used.

    As a concrete example, OpenAI chat targets are PromptChatTargets. You can set made-up conversation history.
    Realtime chat targets or OpenAI completions are NOT PromptChatTargets. You don't send the conversation history.
    """

    def __init__(
        self,
        *,
        max_requests_per_minute: Optional[int] = None,
        endpoint: str = "",
        model_name: str = "",
    ) -> None:
        super().__init__(max_requests_per_minute=max_requests_per_minute, endpoint=endpoint, model_name=model_name)

    def set_system_prompt(
        self,
        *,
        system_prompt: str,
        conversation_id: str,
        attack_identifier: Optional[dict[str, str]] = None,
        labels: Optional[dict[str, str]] = None,
    ) -> None:
        """
        Set the system prompt for the prompt target. May be overridden by subclasses.

        Raises:
            RuntimeError: If the conversation already exists.
        """
        messages = self._memory.get_conversation(conversation_id=conversation_id)

        if messages:
            raise RuntimeError("Conversation already exists, system prompt needs to be set at the beginning")

        self._memory.add_message_to_memory(
            request=MessagePiece(
                role="system",
                conversation_id=conversation_id,
                original_value=system_prompt,
                converted_value=system_prompt,
                prompt_target_identifier=self.get_identifier(),
                attack_identifier=attack_identifier,
                labels=labels,
            ).to_message()
        )

    @abc.abstractmethod
    def is_json_response_supported(self) -> bool:
        """
        Abstract method to determine if JSON response format is supported by the target.

        Returns:
            bool: True if JSON response is supported, False otherwise.
        """
        pass

    def is_response_format_json(self, message_piece: MessagePiece) -> bool:
        """
        Check if the response format is JSON and ensure the target supports it.

        Args:
            message_piece: A MessagePiece object with a `prompt_metadata` dictionary that may
                include a "response_format" key.

        Returns:
            bool: True if the response format is JSON and supported, False otherwise.

        Raises:
            ValueError: If "json" response format is requested but unsupported.
        """
        if message_piece.prompt_metadata:
            response_format = message_piece.prompt_metadata.get("response_format")
            if response_format == "json":
                if not self.is_json_response_supported():
                    target_name = self.get_identifier()["__type__"]
                    raise ValueError(f"This target {target_name} does not support JSON response format.")
                return True
        return False
