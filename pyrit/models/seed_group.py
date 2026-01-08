# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

import logging
import uuid
from collections import defaultdict
from typing import Any, Dict, List, Optional, Sequence, Union

from pyrit.common.yaml_loadable import YamlLoadable
from pyrit.models.message import Message
from pyrit.models.message_piece import MessagePiece
from pyrit.models.seed import Seed
from pyrit.models.seed_objective import SeedObjective
from pyrit.models.seed_prompt import SeedPrompt

logger = logging.getLogger(__name__)


class SeedGroup(YamlLoadable):
    """
    A group of prompts that need to be sent together, along with an objective. This can include multiturn and multimodal
    prompts.
    This class is useful when a target requires multiple message pieces to be grouped and sent together.
    All prompts in the group should share the same `prompt_group_id`.
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

        self.seeds = list([self.objective] if self.objective else []) + list(sorted_prompts)

    @property
    def objective(self) -> Optional[SeedObjective]:
        for seed in self.seeds:
            if isinstance(seed, SeedObjective):
                return seed
        return None

    def set_objective(self, value: str) -> None:
        """
        Set or update the objective for this SeedGroup.

        If an objective already exists, updates its value.
        If not, creates a new SeedObjective and inserts it at the beginning.

        Args:
            value (str): The objective value to set.
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
    def prompts(self) -> Sequence[SeedPrompt]:
        return [seed for seed in self.seeds if isinstance(seed, SeedPrompt)]

    @property
    def harm_categories(self) -> List[str]:
        """
        Returns a flattened list of all harm categories from all seeds in the group.

        Returns:
            List[str]: A list of harm categories from all seeds, with duplicates removed.
        """
        categories: List[str] = []
        for seed in self.seeds:
            if seed.harm_categories:
                categories.extend(seed.harm_categories)
        return list(set(categories))

    def render_template_value(self, **kwargs: object) -> None:
        """
        Renders self.value as a template, applying provided parameters in kwargs.

        Args:
            kwargs:Key-value pairs to replace in the SeedGroup value.

        Returns:
            None

        Raises:
            ValueError: If parameters are missing or invalid in the template.
        """
        for seed in self.seeds:
            seed.value = seed.render_template_value(**kwargs)

    def _enforce_max_one_objective(self) -> None:
        if len([s for s in self.seeds if isinstance(s, SeedObjective)]) > 1:
            raise ValueError("SeedGroups can only have one objective.")

    def _enforce_consistent_group_id(self) -> None:
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

    def _enforce_consistent_role(self) -> None:
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
        grouped_prompts: dict[int, list[SeedPrompt]] = defaultdict(list)
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

    def is_single_turn(self) -> bool:
        return self.is_single_request() and not self.objective

    def is_single_request(self) -> bool:
        unique_sequences = {prompt.sequence for prompt in self.prompts}
        return len(unique_sequences) <= 1

    def is_single_part_single_text_request(self) -> bool:
        return len(self.prompts) == 1 and self.prompts[0].data_type == "text"

    @property
    def prepended_conversation(self) -> Optional[List[Message]]:
        """
        Returns Messages that should be prepended as conversation history.

        If the last message in the sequence is a user message, returns all messages
        except the last one. If the last message is not a user message, returns
        the entire sequence as conversation history.

        Returns:
            Optional[List[Message]]: Messages for conversation history, or None if empty.
        """
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

        If the last message in the sequence is a user message, returns that message.
        If the last message is not a user message, returns None.

        Returns:
            Optional[Message]: Message for the current/last turn if user role, or None otherwise.
        """
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
            List[Message]: All user messages in sequence order, or empty list if no prompts.
        """
        if not self.prompts:
            return []

        return self._prompts_to_messages(list(self.prompts))

    def _get_last_sequence_role(self) -> Optional[str]:
        """
        Get the role of the last sequence in this SeedGroup.

        Returns:
            Optional[str]: The role of the last sequence, or None if no prompts exist.
        """
        if not self.prompts:
            return None

        unique_sequences = sorted({prompt.sequence for prompt in self.prompts})
        last_sequence = unique_sequences[-1]
        last_sequence_prompts = [p for p in self.prompts if p.sequence == last_sequence]

        # Role should be consistent within a sequence (enforced by _enforce_consistent_role)
        return last_sequence_prompts[0].role if last_sequence_prompts else None

    def _prompts_to_messages(self, prompts: Sequence[SeedPrompt]) -> List[Message]:
        """
        Convert a sequence of SeedPrompts to Messages.

        Groups prompts by sequence number and creates one Message per sequence.

        Args:
            prompts (Sequence[SeedPrompt]): The prompts to convert.

        Returns:
            List[Message]: Messages created from the prompts.
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

    def __repr__(self) -> str:
        return f"<SeedGroup(seeds={len(self.seeds)} seeds)>"
