# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from dataclasses import dataclass, field
from typing import List, Literal, Optional

from pyrit.message_normalizer import MessageStringNormalizer
from pyrit.models import ChatMessageRole


@dataclass
class PrependedConversationConfiguration:
    """
    Configuration for controlling how prepended conversations are processed before
    being sent to targets.

    This class provides control over:
    - Which message roles should have request converters applied
    - How to normalize/squash conversation history into a single text block
    - How to handle text insertion when the first message piece is non-text
    - What to do when the target is not a PromptChatTarget
    """

    # Roles for which request converters should be applied to prepended messages.
    # By default, no converters are applied (empty list).
    # Example: ["user"] to apply converters only to user messages.
    apply_converters_to_roles: List[ChatMessageRole] = field(default_factory=list)

    # Optional normalizer to squash conversation history into a single text block.
    # Must implement MessageStringNormalizer (e.g., ChatMLNormalizer or ConversationContextNormalizer).
    # When provided, the prepended conversation is normalized using this normalizer.
    # When None and normalization is needed (e.g., for non-chat targets), a default
    # ConversationContextNormalizer is used that produces a readable "Turn N: User/Assistant" format.
    message_normalizer: Optional[MessageStringNormalizer] = None

    # Controls text insertion behavior when message_normalizer is used:
    # - True: Prepend normalized text to the first text piece if one exists.
    #   If no text piece exists, insert a new text piece at the beginning.
    # - False: Always insert a new text piece at the beginning, even if a text piece exists.
    prepend_to_first_text_piece: bool = True

    # Behavior when the target is a PromptTarget but not a PromptChatTarget:
    # - "normalize_first_turn": Normalize the prepended conversation into a string and
    #   include it in the first message. Uses message_normalizer if provided, otherwise
    #   falls back to a default "Turn N: User/Assistant" format. This allows using
    #   prepended conversation context with non-chat targets that don't maintain history.
    # - "raise": Raise a ValueError. Use this when prepended conversation history must be
    #   maintained by the target (i.e., target must be a PromptChatTarget).
    non_chat_target_behavior: Literal["normalize_first_turn", "raise"] = "normalize_first_turn"

    @classmethod
    def default(cls) -> "PrependedConversationConfiguration":
        """
        Create a default configuration with no converters applied and no normalization.

        Returns:
            A configuration that passes prepended conversation through without
            modification, raising an error for non-chat targets.
        """
        return cls(
            apply_converters_to_roles=[],
            message_normalizer=None,
            non_chat_target_behavior="raise",
        )

    @classmethod
    def with_normalizer(
        cls,
        *,
        normalizer: MessageStringNormalizer,
        apply_converters_to_roles: Optional[List[ChatMessageRole]] = None,
        prepend_to_first_text_piece: bool = True,
    ) -> "PrependedConversationConfiguration":
        """
        Create a configuration that normalizes prepended conversation into a text block.

        This is useful for targets that don't support conversation history natively,
        allowing the context to be included in the first message.

        Args:
            normalizer: A MessageStringNormalizer (e.g., ChatMLNormalizer, ConversationContextNormalizer)
                to format the conversation into a single text block.
            apply_converters_to_roles: Roles to apply converters to before normalization.
                Defaults to empty list (no converters applied).
            prepend_to_first_text_piece: Whether to prepend to an existing text piece
                or insert a new one. Defaults to True.

        Returns:
            A configuration that normalizes the prepended conversation and handles
            non-chat targets gracefully.
        """
        return cls(
            apply_converters_to_roles=apply_converters_to_roles or [],
            message_normalizer=normalizer,
            prepend_to_first_text_piece=prepend_to_first_text_piece,
            non_chat_target_behavior="normalize_first_turn",
        )
