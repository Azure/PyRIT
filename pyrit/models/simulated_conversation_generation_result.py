# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
SimulatedConversationGenerationResult - Result from generating a simulated conversation.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class SimulatedConversationGenerationResult:
    """
    Result from generating a simulated conversation.

    This contains the generated messages ready for use as prepended_conversation
    and next_message, along with metadata about the generation.
    """

    # Messages to use as prepended_conversation (all but last turn)
    prepended_messages: List[Any]  # List[Message] - using Any to avoid circular import

    # The next message to send (last user turn)
    next_message: Optional[Any]  # Optional[Message]

    # Score from evaluating the final turn
    score: Optional[Any]  # Optional[Score]

    # Identifier capturing the configuration used for generation
    identifier: Dict[str, Any]

    # Full conversation (for debugging/inspection)
    full_conversation: List[Any]  # List[Message]
