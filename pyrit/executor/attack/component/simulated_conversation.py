# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
Utility functions for generating simulated conversations using adversarial chat.

These utilities help create prepended_conversation content by running an adversarial chat
against a simulated (compliant) target before executing the actual attack.
"""

from __future__ import annotations

import enum
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Union

from pyrit.common.path import EXECUTOR_SIMULATED_TARGET_PATH
from pyrit.executor.attack.core import (
    AttackAdversarialConfig,
    AttackConverterConfig,
    AttackScoringConfig,
)
from pyrit.executor.attack.multi_turn.red_teaming import RedTeamingAttack
from pyrit.memory import CentralMemory
from pyrit.models import Message, Score, SeedPrompt
from pyrit.prompt_target import PromptChatTarget
from pyrit.score import TrueFalseScorer

logger = logging.getLogger(__name__)


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

    conversation: List[Message]
    score: Optional[Score]
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
        if len(self.conversation) % 2 == 1 and self.conversation[-1].role == "user":
            total_turns += 1

        if self.turn_index is None:
            return total_turns
        return max(1, min(self.turn_index, total_turns))

    @property
    def prepended_messages(self) -> List[Message]:
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
    def next_message(self) -> Optional[Message]:
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
        if user_idx < len(self.conversation) and self.conversation[user_idx].role == "user":
            return self.conversation[user_idx].duplicate_message()
        return None


class SimulatedTargetSystemPromptPaths(enum.Enum):
    """Enum for predefined simulated target system prompt paths."""

    COMPLIANT = Path(EXECUTOR_SIMULATED_TARGET_PATH, "compliant.yaml").resolve()


async def generate_simulated_conversation_async(
    *,
    objective: str,
    adversarial_chat: PromptChatTarget,
    objective_scorer: TrueFalseScorer,
    num_turns: int = 3,
    adversarial_chat_system_prompt_path: Union[str, Path],
    simulated_target_system_prompt_path: Optional[Union[str, Path]] = None,
    attack_converter_config: Optional[AttackConverterConfig] = None,
    memory_labels: Optional[dict[str, str]] = None,
) -> SimulatedConversationResult:
    """
    Generate a simulated conversation between an adversarial chat and a compliant target.

    This utility runs a RedTeamingAttack with `score_last_turn_only=True` against a simulated
    target (the same LLM as adversarial_chat, but configured with a compliant system prompt).
    The resulting conversation can be used as `prepended_conversation` for subsequent attacks
    against real targets.

    Use cases:
    - Creating role-play scenarios dynamically (e.g., movie script, video game)
    - Establishing conversational context before attacking a real target
    - Generating multi-turn jailbreak setups without hardcoded responses

    Args:
        objective (str): The objective for the adversarial chat to work toward.
        adversarial_chat (PromptChatTarget): The adversarial LLM that generates attack prompts.
            This same LLM is also used as the simulated target with a compliant system prompt.
        objective_scorer (TrueFalseScorer): Scorer to evaluate the final turn.
        num_turns (int): Number of conversation turns to generate. Defaults to 3.
        adversarial_chat_system_prompt_path (Union[str, Path]): Path to the system prompt
            for the adversarial chat. This is required.
        simulated_target_system_prompt_path (Optional[Union[str, Path]]): Path to the system prompt
            for the simulated target. If not provided, uses the default compliant prompt.
            The template should accept `objective` and `num_turns` parameters.
        attack_converter_config (Optional[AttackConverterConfig]): Converter configuration for
            the attack. Defaults to None.
        memory_labels (Optional[dict[str, str]]): Labels to associate with the conversation
            in memory. Defaults to None.

    Returns:
        SimulatedConversationResult: The result containing the generated conversation and score.
            Use `prepended_messages` to get completed turns before the selected turn,
            `next_message` to get the user message at the selected turn for use as an
            attack's initial prompt, or access `conversation` directly for all messages.
            Set `turn_index` to select an earlier turn if the final turn wasn't successful.

    Raises:
        ValueError: If num_turns is not a positive integer.
    """
    # Use the same LLM for both adversarial chat and simulated target
    # They get different system prompts to play different roles
    simulated_target = adversarial_chat
    if num_turns <= 0:
        raise ValueError("num_turns must be a positive integer")

    # Load and configure simulated target system prompt
    simulated_target_prompt_path = (
        simulated_target_system_prompt_path or SimulatedTargetSystemPromptPaths.COMPLIANT.value
    )
    simulated_target_system_prompt_template = SeedPrompt.from_yaml_with_required_parameters(
        template_path=simulated_target_prompt_path,
        required_parameters=["objective", "num_turns"],
        error_message="Simulated target system prompt must have objective and num_turns parameters",
    )
    simulated_target_system_prompt = simulated_target_system_prompt_template.render_template_value(
        objective=objective,
        num_turns=num_turns,
    )

    # Create adversarial config for the simulation
    adversarial_config = AttackAdversarialConfig(
        target=adversarial_chat,
        system_prompt_path=adversarial_chat_system_prompt_path,
    )

    # Create scoring config
    scoring_config = AttackScoringConfig(
        objective_scorer=objective_scorer,
        use_score_as_feedback=False,  # Don't need feedback for last-turn-only scoring
    )

    # Create the RedTeamingAttack with simulated target and score_last_turn_only
    attack = RedTeamingAttack(
        objective_target=simulated_target,
        attack_adversarial_config=adversarial_config,
        attack_converter_config=attack_converter_config,
        attack_scoring_config=scoring_config,
        max_turns=num_turns,
        score_last_turn_only=True,
    )

    # Execute the simulated attack
    logger.info(f"Generating {num_turns}-turn simulated conversation for objective: {objective[:50]}...")

    # Create a system message to prepend - this sets the simulated target's behavior
    system_message = Message.from_system_prompt(simulated_target_system_prompt)

    result = await attack.execute_async(
        objective=objective,
        prepended_conversation=[system_message],
        memory_labels=memory_labels,
    )

    # Extract the conversation from memory and filter for prepended_conversation use
    memory = CentralMemory.get_memory_instance()
    raw_messages = list(memory.get_conversation(conversation_id=result.conversation_id))

    # Filter out system messages - prepended_conversation should only have user/assistant turns
    # System prompts are set separately on each target during attack execution
    filtered_messages: List[Message] = []
    for message in raw_messages:
        if message.role != "system":
            filtered_messages.append(message)

    # Get the score from the result (there should be one score for the last turn)
    final_score = result.last_score

    logger.info(
        f"Generated simulated conversation with {len(filtered_messages)} messages " f"(outcome: {result.outcome.name})"
    )

    return SimulatedConversationResult(
        conversation=filtered_messages,
        score=final_score,
    )
