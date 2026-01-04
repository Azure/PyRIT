# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
Utility functions for generating simulated conversations using adversarial chat.

These utilities help create prepended_conversation content by running an adversarial chat
against a simulated (compliant) target before executing the actual attack.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Union

from pyrit.executor.attack.core import (
    AttackAdversarialConfig,
    AttackConverterConfig,
    AttackScoringConfig,
)
from pyrit.executor.attack.multi_turn.red_teaming import RedTeamingAttack
from pyrit.memory import CentralMemory
from pyrit.models import Message, SimulatedConversationResult
from pyrit.models.seeds import SeedSimulatedConversation
from pyrit.prompt_target import PromptChatTarget
from pyrit.score import TrueFalseScorer

logger = logging.getLogger(__name__)


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

    # Load and configure simulated target system prompt using centralized validation
    simulated_target_system_prompt = SeedSimulatedConversation.load_simulated_target_system_prompt(
        objective=objective,
        num_turns=num_turns,
        simulated_target_system_prompt_path=simulated_target_system_prompt_path,
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
    # Also mark assistant messages as simulated for traceability
    filtered_messages: List[Message] = []
    for message in raw_messages:
        if message.api_role != "system":
            # Mark assistant responses as simulated since this is a simulated conversation
            if message.api_role == "assistant":
                for piece in message.message_pieces:
                    piece._role = "simulated_assistant"
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
