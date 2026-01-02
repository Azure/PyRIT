# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
SeedGroup and SeedAttackGroup classes for grouping seeds together.

SeedGroup is a base container for grouping prompts together.
SeedAttackGroup extends SeedGroup with attack-specific functionality including
objectives, prepended conversations, and simulated conversation configuration.
"""

from __future__ import annotations

import logging
import uuid
import warnings
from collections import defaultdict
from typing import Any, Dict, List, Optional, Sequence, Union

from pyrit.common.yaml_loadable import YamlLoadable
from pyrit.models.message import Message, MessagePiece
from pyrit.models.seeds.seed import Seed
from pyrit.models.seeds.seed_objective import SeedObjective
from pyrit.models.seeds.seed_prompt import SeedPrompt
from pyrit.models.seeds.seed_simulated_conversation import SeedSimulatedConversation
from pyrit.models.simulated_conversation_generation_result import SimulatedConversationGenerationResult

logger = logging.getLogger(__name__)


class SeedGroup(YamlLoadable):
    """
    A base container for grouping prompts that need to be sent together.

    This class is useful when a target requires multiple message pieces to be grouped
    and sent together. All prompts in the group should share the same `prompt_group_id`.

    For attack-specific functionality (objectives, prepended conversations, etc.),
    use SeedAttackGroup instead.
    """

    seeds: List[Seed]

    def __init__(
        self,
        *,
        seeds: Sequence[Union[Seed, Dict[str, Any]]],
    ):
        if not seeds:
            raise ValueError("SeedGroup cannot be empty.")

        self.seeds = []
        for seed in seeds:
            if isinstance(seed, SeedObjective):
                self.seeds.append(seed)
            elif isinstance(seed, SeedPrompt):
                self.seeds.append(seed)
            elif isinstance(seed, dict):
                # create a SeedObjective in addition to the SeedPrompt if is_objective is True
                is_objective = seed.pop("is_objective", False)
                if is_objective:
                    self.seeds.append(SeedObjective(**seed))
                else:
                    self.seeds.append(SeedPrompt(**seed))
            else:
                raise ValueError(f"Invalid seed type: {type(seed)}")

        self._enforce_consistent_group_id()
        self._enforce_consistent_role()
        self._enforce_max_one_objective()

        sorted_prompts = sorted(self.prompts, key=lambda prompt: prompt.sequence if prompt.sequence is not None else 0)

        self.seeds = list([self._get_objective()] if self._get_objective() else []) + list(sorted_prompts)

    def _get_objective(self) -> Optional[SeedObjective]:
        """Get the objective seed if present."""
        for seed in self.seeds:
            if isinstance(seed, SeedObjective):
                return seed
        return None

    @property
    def prompts(self) -> Sequence[SeedPrompt]:
        """Get all SeedPrompt instances from this group."""
        return [seed for seed in self.seeds if isinstance(seed, SeedPrompt)]

    @property
    def harm_categories(self) -> List[str]:
        """
        Returns a flattened list of all harm categories from all seeds in the group.

        Returns:
            A list of harm categories from all seeds, with duplicates removed.
        """
        categories: List[str] = []
        for seed in self.seeds:
            if seed.harm_categories:
                categories.extend(seed.harm_categories)
        return list(set(categories))

    def render_template_value(self, **kwargs):
        """
        Renders self.value as a template, applying provided parameters in kwargs.

        Args:
            kwargs: Key-value pairs to replace in the SeedGroup value.
        """
        for seed in self.seeds:
            seed.value = seed.render_template_value(**kwargs)

    def _enforce_max_one_objective(self):
        if len([s for s in self.seeds if isinstance(s, SeedObjective)]) > 1:
            raise ValueError("SeedGroups can only have one objective.")

    def _enforce_consistent_group_id(self):
        """
        Ensures that if any of the seeds already have a group ID set,
        they share the same ID. If none have a group ID set, assign a
        new UUID to all seeds.

        Raises:
            ValueError: If multiple different group IDs exist among the seeds.
        """
        existing_group_ids = {seed.prompt_group_id for seed in self.seeds if seed.prompt_group_id is not None}

        if len(existing_group_ids) > 1:
            # More than one distinct group ID found among seeds.
            raise ValueError("Inconsistent group IDs found across seeds.")
        elif len(existing_group_ids) == 1:
            # Exactly one group ID is set; apply it to all.
            group_id = existing_group_ids.pop()
            for seed in self.seeds:
                seed.prompt_group_id = group_id
        else:
            # No group IDs set; generate a fresh one and assign it to all.
            new_group_id = uuid.uuid4()
            for seed in self.seeds:
                seed.prompt_group_id = new_group_id

    def _enforce_consistent_role(self):
        """
        Ensures that all prompts in the group that share a sequence have a consistent role.
        If no roles are set, all prompts will be assigned the default 'user' role.
        If only one prompt in a sequence has a role, all prompts will be assigned that role.
        Roles must be set if there is more than one sequence and within a sequence all
        roles must be consistent.

        Raises:
            ValueError: If multiple different roles are found across prompts in the group  or
            if no roles are set in a multi-sequence group.
        """
        # groups the prompts according to their sequence
        grouped_prompts = defaultdict(list)
        for prompt in self.prompts:
            if prompt.sequence not in grouped_prompts:
                grouped_prompts[prompt.sequence] = []
            grouped_prompts[prompt.sequence].append(prompt)

        num_sequences = len(grouped_prompts)
        for sequence, prompts in grouped_prompts.items():
            roles = {prompt.role for prompt in prompts if prompt.role is not None}
            if not len(roles) and num_sequences > 1:
                raise ValueError(
                    f"No roles set for sequence {sequence} in a multi-sequence group. "
                    "Please ensure at least one prompt within a sequence has an assigned role."
                )
            if len(roles) > 1:
                raise ValueError(f"Inconsistent roles found for sequence {sequence}: {roles}")
            role = roles.pop() if len(roles) else "user"
            for prompt in prompts:
                prompt.role = role

    def _prompts_to_messages(self, prompts: Sequence[SeedPrompt]) -> List[Message]:
        """
        Convert a sequence of SeedPrompts to Messages.

        Groups prompts by sequence number and creates one Message per sequence.

        Args:
            prompts: The prompts to convert.

        Returns:
            Messages created from the prompts.
        """
        # Group by sequence
        sequence_groups = defaultdict(list)
        for prompt in prompts:
            sequence_groups[prompt.sequence].append(prompt)

        messages = []
        for sequence in sorted(sequence_groups.keys()):
            sequence_prompts = sequence_groups[sequence]

            # Convert each prompt to a MessagePiece
            message_pieces = []
            for prompt in sequence_prompts:
                # Convert assistant to simulated_assistant for YAML-loaded conversations
                # since these represent simulated/prepended content, not actual target responses
                role = prompt.role or "user"
                if role == "assistant":
                    role = "simulated_assistant"

                piece = MessagePiece(
                    role=role,
                    original_value=prompt.value,
                    original_value_data_type=prompt.data_type or "text",
                    prompt_target_identifier=None,
                    conversation_id=str(prompt.prompt_group_id),
                    sequence=sequence,
                    prompt_metadata=prompt.metadata,
                )
                message_pieces.append(piece)

            # Create Message from pieces
            messages.append(Message(message_pieces=message_pieces))

        return messages

    def __repr__(self):
        return f"<SeedGroup(seeds={len(self.seeds)} seeds)>"


class SeedAttackGroup(SeedGroup):
    """
    A group of seeds for use in attack scenarios, with an objective and optional
    prepended conversation or simulated conversation configuration.

    This class extends SeedGroup with attack-specific functionality:
    - Required objective for attack goals
    - Prepended conversation support (from prompts or externally generated)
    - Next message extraction for attack initialization
    - Simulated conversation configuration (generation happens externally)

    The simulated_conversation field holds configuration for dynamic generation of
    prepended conversations. The actual generation is performed by the executor layer
    (e.g., `generate_simulated_conversation_async` in simulated_conversation.py).
    Results are stored via `set_simulated_conversation_result()`.
    """

    def __init__(
        self,
        *,
        seeds: Sequence[Union[Seed, Dict[str, Any]]],
        simulated_conversation: Optional[SeedSimulatedConversation] = None,
    ):
        """
        Initialize a SeedAttackGroup.

        Args:
            seeds: Sequence of seeds (must include at least one SeedObjective).
            simulated_conversation: Optional configuration for generating prepended
                conversations dynamically. Cannot be used with multi-sequence prompts.

        Raises:
            ValueError: If seeds is empty or contains invalid types.
            ValueError: If simulated_conversation is set with multi-sequence prompts.
        """
        super().__init__(seeds=seeds)

        self._simulated_conversation_config = simulated_conversation
        self._cached_simulated_result: Optional[SimulatedConversationGenerationResult] = None

        # Validate mutual exclusivity
        if simulated_conversation is not None and self._has_multi_sequence_prompts():
            raise ValueError(
                "Cannot use simulated_conversation with multi-sequence prompts. "
                "The simulated conversation generates the prepended conversation dynamically."
            )

    def _has_multi_sequence_prompts(self) -> bool:
        """Check if prompts span multiple sequences (would create prepended_conversation)."""
        unique_sequences = {prompt.sequence for prompt in self.prompts}
        return len(unique_sequences) > 1

    @property
    def objective(self) -> Optional[SeedObjective]:
        """Get the objective for this attack group."""
        return self._get_objective()

    def set_objective(self, value: str) -> None:
        """
        Set or update the objective for this SeedAttackGroup.

        If an objective already exists, updates its value.
        If not, creates a new SeedObjective and inserts it at the beginning.

        Args:
            value: The objective value to set.
        """
        if self.objective is not None:
            self.objective.value = value
        else:
            new_objective = SeedObjective(value=value)
            # Match the group ID from existing seeds
            if self.seeds:
                new_objective.prompt_group_id = self.seeds[0].prompt_group_id
            self.seeds.insert(0, new_objective)

    @property
    def simulated_conversation_config(self) -> Optional[SeedSimulatedConversation]:
        """Get the simulated conversation configuration if set."""
        return self._simulated_conversation_config

    @property
    def has_simulated_conversation(self) -> bool:
        """Check if this group uses simulated conversation generation."""
        return self._simulated_conversation_config is not None

    @property
    def simulated_conversation_generated(self) -> bool:
        """Check if the simulated conversation has been generated and cached."""
        return self._cached_simulated_result is not None

    def set_simulated_conversation_result(
        self, result: SimulatedConversationGenerationResult
    ) -> None:
        """
        Store the result of simulated conversation generation.

        This method is called by the executor layer after generating the simulated
        conversation. It caches the result for use by prepended_conversation and next_message.

        Args:
            result: The generation result from the executor.
        """
        self._cached_simulated_result = result

    def clear_cached_simulated_conversation(self) -> None:
        """Clear the cached simulated conversation result."""
        self._cached_simulated_result = None

    @property
    def simulated_conversation_identifier(self) -> Optional[Dict[str, Any]]:
        """
        Get the identifier for the cached simulated conversation.

        Returns:
            The identifier dict if a simulated conversation has been generated, None otherwise.
        """
        if self._cached_simulated_result is not None:
            return self._cached_simulated_result.identifier
        return None

    @property
    def prepended_conversation(self) -> Optional[List[Message]]:
        """
        Returns Messages that should be prepended as conversation history.

        If a simulated conversation has been generated, returns those messages.
        Otherwise, if the last message in the sequence is a user message, returns all
        messages except the last one. If the last message is not a user message,
        returns the entire sequence as conversation history.

        Returns:
            Messages for conversation history, or None if empty.
        """
        # Check for cached simulated conversation first
        if self._cached_simulated_result is not None:
            return self._cached_simulated_result.prepended_messages or None

        # Fall back to prompt-based prepended conversation
        if not self.prompts:
            return None

        last_role = self._get_last_sequence_role()
        unique_sequences = sorted({prompt.sequence for prompt in self.prompts})

        if last_role == "user":
            # Last message is user - prepend everything except the last sequence
            if len(unique_sequences) <= 1:
                return None

            last_sequence = unique_sequences[-1]
            prepended_prompts = [p for p in self.prompts if p.sequence != last_sequence]

            if not prepended_prompts:
                return None

            return self._prompts_to_messages(prepended_prompts)
        else:
            # Last message is not user - entire sequence is prepended conversation
            return self._prompts_to_messages(list(self.prompts))

    @property
    def next_message(self) -> Optional[Message]:
        """
        Returns a Message containing only the last turn's prompts if it's a user message.

        If a simulated conversation has been generated, returns the next_message from that.
        Otherwise, if the last message in the sequence is a user message, returns that message.
        If the last message is not a user message, returns None.

        Returns:
            Message for the current/last turn if user role, or None otherwise.
        """
        # Check for cached simulated conversation first
        if self._cached_simulated_result is not None:
            return self._cached_simulated_result.next_message

        # Fall back to prompt-based next_message
        if not self.prompts:
            return None

        last_role = self._get_last_sequence_role()

        if last_role != "user":
            # Last message is not a user message - no next_message
            return None

        unique_sequences = sorted({prompt.sequence for prompt in self.prompts})
        last_sequence = unique_sequences[-1]
        current_turn_prompts = [p for p in self.prompts if p.sequence == last_sequence]

        if not current_turn_prompts:
            return None

        messages = self._prompts_to_messages(current_turn_prompts)
        return messages[0] if messages else None

    @property
    def user_messages(self) -> List[Message]:
        """
        Returns all prompts as user Messages, one per sequence.

        This is used by MultiPromptSendingAttack to get user messages for multi-turn attacks.
        Only returns messages from prompts (not objectives).

        Returns:
            All user messages in sequence order, or empty list if no prompts.
        """
        if not self.prompts:
            return []

        return self._prompts_to_messages(list(self.prompts))

    def _get_last_sequence_role(self) -> Optional[str]:
        """
        Get the role of the last sequence in this SeedGroup.

        Returns:
            The role of the last sequence, or None if no prompts exist.
        """
        if not self.prompts:
            return None

        unique_sequences = sorted({prompt.sequence for prompt in self.prompts})
        last_sequence = unique_sequences[-1]
        last_sequence_prompts = [p for p in self.prompts if p.sequence == last_sequence]

        # Role should be consistent within a sequence (enforced by _enforce_consistent_role)
        return last_sequence_prompts[0].role if last_sequence_prompts else None

    def is_single_turn(self) -> bool:
        """Check if this is a single-turn group (single request without objective)."""
        return self.is_single_request() and not self.objective

    def is_single_request(self) -> bool:
        """Check if all prompts are in a single sequence."""
        unique_sequences = {prompt.sequence for prompt in self.prompts}
        return len(unique_sequences) <= 1

    def is_single_part_single_text_request(self) -> bool:
        """Check if this is a single text prompt."""
        return len(self.prompts) == 1 and self.prompts[0].data_type == "text"

    def __repr__(self):
        sim_info = " (simulated)" if self.has_simulated_conversation else ""
        cached_info = " [cached]" if self.simulated_conversation_generated else ""
        return f"<SeedAttackGroup(seeds={len(self.seeds)}{sim_info}{cached_info})>"


# =============================================================================
# Backward Compatibility
# =============================================================================

# The old SeedGroup class included attack-specific properties (objective,
# prepended_conversation, next_message, user_messages). For backward compatibility,
# we keep these on SeedGroup but issue deprecation warnings when used.

# Store original SeedGroup for internal use
_BaseSeedGroup = SeedGroup


class SeedGroup(_BaseSeedGroup):
    """
    A group of prompts that need to be sent together, along with an objective.

    .. deprecated::
        Attack-specific properties (objective, prepended_conversation, next_message,
        user_messages) are deprecated on SeedGroup. Use SeedAttackGroup instead.

    This class maintains backward compatibility by delegating to SeedAttackGroup
    for attack-specific functionality.
    """

    def __init__(
        self,
        *,
        seeds: Sequence[Union[Seed, Dict[str, Any]]],
    ):
        super().__init__(seeds=seeds)
        # Create an internal SeedAttackGroup for backward-compatible property access
        self._attack_group: Optional[SeedAttackGroup] = None

    def _get_attack_group(self) -> SeedAttackGroup:
        """Lazily create a SeedAttackGroup wrapper for backward compatibility."""
        if self._attack_group is None:
            # Create attack group with same seeds
            self._attack_group = SeedAttackGroup.__new__(SeedAttackGroup)
            self._attack_group.seeds = self.seeds
            self._attack_group._simulated_conversation_config = None
            self._attack_group._cached_simulated_result = None
        return self._attack_group

    @property
    def objective(self) -> Optional[SeedObjective]:
        """
        Get the objective for this group.

        .. deprecated::
            Use SeedAttackGroup.objective instead.
        """
        warnings.warn(
            "SeedGroup.objective is deprecated. Use SeedAttackGroup instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self._get_attack_group().objective

    def set_objective(self, value: str) -> None:
        """
        Set or update the objective for this SeedGroup.

        .. deprecated::
            Use SeedAttackGroup.set_objective instead.

        Args:
            value: The objective value to set.
        """
        warnings.warn(
            "SeedGroup.set_objective is deprecated. Use SeedAttackGroup instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        self._get_attack_group().set_objective(value)

    @property
    def prepended_conversation(self) -> Optional[List[Message]]:
        """
        Returns Messages that should be prepended as conversation history.

        .. deprecated::
            Use SeedAttackGroup.prepended_conversation instead.
        """
        warnings.warn(
            "SeedGroup.prepended_conversation is deprecated. Use SeedAttackGroup instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self._get_attack_group().prepended_conversation

    @property
    def next_message(self) -> Optional[Message]:
        """
        Returns a Message containing only the last turn's prompts if it's a user message.

        .. deprecated::
            Use SeedAttackGroup.next_message instead.
        """
        warnings.warn(
            "SeedGroup.next_message is deprecated. Use SeedAttackGroup instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self._get_attack_group().next_message

    @property
    def user_messages(self) -> List[Message]:
        """
        Returns all prompts as user Messages, one per sequence.

        .. deprecated::
            Use SeedAttackGroup.user_messages instead.
        """
        warnings.warn(
            "SeedGroup.user_messages is deprecated. Use SeedAttackGroup instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self._get_attack_group().user_messages

    def is_single_turn(self) -> bool:
        """
        Check if this is a single-turn group.

        .. deprecated::
            Use SeedAttackGroup.is_single_turn instead.
        """
        warnings.warn(
            "SeedGroup.is_single_turn is deprecated. Use SeedAttackGroup instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self._get_attack_group().is_single_turn()

    def is_single_request(self) -> bool:
        """
        Check if all prompts are in a single sequence.

        .. deprecated::
            Use SeedAttackGroup.is_single_request instead.
        """
        warnings.warn(
            "SeedGroup.is_single_request is deprecated. Use SeedAttackGroup instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self._get_attack_group().is_single_request()

    def is_single_part_single_text_request(self) -> bool:
        """
        Check if this is a single text prompt.

        .. deprecated::
            Use SeedAttackGroup.is_single_part_single_text_request instead.
        """
        warnings.warn(
            "SeedGroup.is_single_part_single_text_request is deprecated. Use SeedAttackGroup instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self._get_attack_group().is_single_part_single_text_request()
