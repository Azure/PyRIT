# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
SimulatedConversationResult - Result from generating a simulated conversation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Optional

if TYPE_CHECKING:
    from pyrit.models import Message, Score


@dataclass
class SimulatedConversationResult:
    """
    Result from generating a simulated conversation.

    Stores the full conversation and provides properties to access different views of it
    for various attack strategy use cases.

    The conversation attribute contains the complete conversation as a list of Messages
    (user/assistant only, no system messages). The score attribute holds the score from
    evaluating the final turn. The turn_index is a 1-based index of the turn to treat as
    the "final" turn for splitting. If None (default), uses the last turn. Can be set after
    creation to select an earlier turn (e.g., if the last turn's attack didn't work).
    """

    conversation: List["Message"]
    score: Optional["Score"]
    turn_index: Optional[int] = None

    @property
    def _effective_turn_index(self) -> int:
        """
        Get the effective 1-based turn index.

        Returns:
            int: The turn index to use, bounded by available turns.
        """
        if not self.conversation:
            return 0
        # Calculate total complete turns (user+assistant pairs)
        total_turns = len(self.conversation) // 2
        # Account for trailing user message (incomplete turn)
        if len(self.conversation) % 2 == 1 and self.conversation[-1].api_role == "user":
            total_turns += 1

        if self.turn_index is None:
            return total_turns
        return max(1, min(self.turn_index, total_turns))

    @property
    def prepended_messages(self) -> List["Message"]:
        """
        Get all messages before the selected turn with new IDs.

        This returns completed turns before the turn specified by `turn_index`,
        suitable for use as `prepended_conversation` in attack strategies.
        Each message is duplicated with new IDs to avoid database conflicts
        when the messages are inserted into memory by a subsequent attack.

        Returns:
            List[Message]: All messages before the selected turn with fresh IDs.
        """
        turn = self._effective_turn_index
        if turn <= 1:
            return []
        # Each complete turn is 2 messages (user + assistant)
        # Messages before turn N: first (N-1) * 2 messages
        messages = self.conversation[: (turn - 1) * 2]
        return [msg.duplicate_message() for msg in messages]

    @property
    def next_message(self) -> Optional["Message"]:
        """
        Get the user message at the selected turn with a new ID.

        This is the user message from the turn specified by `turn_index`, which
        can be used as the initial prompt/next_message for an attack strategy.
        The message is duplicated with a new ID to avoid database conflicts.

        Returns:
            Optional[Message]: The user message at the selected turn with a fresh ID, or None if not found.
        """
        turn = self._effective_turn_index
        if turn < 1:
            return None
        # User message for turn N is at index (N-1) * 2
        user_idx = (turn - 1) * 2
        if user_idx < len(self.conversation) and self.conversation[user_idx].api_role == "user":
            return self.conversation[user_idx].duplicate_message()
        return None
