# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

import logging
import uuid
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Union

from pyrit.common.yaml_loadable import YamlLoadable
from pyrit.models.seed import Seed
from pyrit.models.seed_objective import SeedObjective
from pyrit.models.seed_prompt import SeedPrompt

from pyrit.models.message import Message, MessagePiece

logger = logging.getLogger(__name__)


@dataclass
class DecomposedSeedGroup:
    """
    A decomposed representation of a SeedGroup, useful for attacks that expect specific arguments.
    
    This structure separates a SeedGroup into its constituent parts:
    - The objective as a string (if present)
    - Prepended conversation history (all turns except the last)
    - Current turn seed group (only the last turn)
    """

    # The objective as a string, if a SeedObjective exists in the original group
    objective: Optional[str] = None

    # Messages representing prior conversation turns (excludes the current/last turn)
    prepended_conversation: Optional[List["Message"]] = None

    # SeedGroup containing only SeedPrompts from the last turn
    current_turn_seed_group: Optional["SeedGroup"] = None


class SeedGroup(YamlLoadable):
    """
    A group of prompts that need to be sent together, along with an objective. This can include multiturn and multimodal
    prompts.
    This class is useful when a target requires multiple message pieces to be grouped and sent together.
    All prompts in the group should share the same `prompt_group_id`.
    """

    seeds: Sequence[Seed]

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
        
        sorted_prompts = sorted(
            self.prompts, key=lambda prompt: prompt.sequence if prompt.sequence is not None else 0
        )
        
        self.seeds = ([self.objective] if self.objective else []) + sorted_prompts

    @property
    def objective(self) -> Optional[SeedObjective]:
        for seed in self.seeds:
            if isinstance(seed, SeedObjective):
                return seed
        return None

    @property
    def prompts(self) -> Sequence[SeedPrompt]:
        return [seed for seed in self.seeds if isinstance(seed, SeedPrompt)]

    def render_template_value(self, **kwargs):
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




    def is_single_turn(self) -> bool:
        return self.is_single_request() and not self.objective

    def is_single_request(self) -> bool:
        unique_sequences = {prompt.sequence for prompt in self.prompts}
        return len(unique_sequences) <= 1

    def is_single_part_single_text_request(self) -> bool:
        return len(self.prompts) == 1 and self.prompts[0].data_type == "text"

    def to_attack_parameters(self) -> DecomposedSeedGroup:
        """
        Decomposes this SeedGroup into parts commonly used by attack strategies.
        
        This method extracts:
        - objective: The string value from SeedObjective (if present)
        - prepended_conversation: Messages from all sequences except the last turn
        - current_turn_seed_group: A new SeedGroup containing only the last turn's prompts
        
        Returns:
            DecomposedSeedGroup: Object containing the decomposed parts.
        
        Raises:
            ValueError: If conversion to Messages fails.
        """
        from pyrit.models.message import Message
        from pyrit.models.message_piece import MessagePiece
        
        result = DecomposedSeedGroup()
        
        # Extract objective if present
        if self.objective:
            result.objective = self.objective.value
        
        # Get unique sequences and determine if we have multiple turns
        unique_sequences = sorted({prompt.sequence for prompt in self.prompts if prompt.sequence is not None})
        
        if len(unique_sequences) > 1:
            # Multiple turns exist - split into prepended and current
            last_sequence = unique_sequences[-1]
            
            # Create prepended_conversation from all but the last sequence
            prepended_prompts = [p for p in self.prompts if p.sequence != last_sequence]
            if prepended_prompts:
                result.prepended_conversation = self._prompts_to_messages(prepended_prompts)
            
            # Create current_turn_seed_group from last sequence only
            current_turn_prompts = [p for p in self.prompts if p.sequence == last_sequence]
            if current_turn_prompts:
                result.current_turn_seed_group = SeedGroup(seeds=current_turn_prompts)
        else:
            # Single turn - everything goes to current_turn_seed_group
            result.current_turn_seed_group = SeedGroup(seeds=list(self.prompts))
        
        return result
    
    def _prompts_to_messages(self, prompts: Sequence[SeedPrompt]) -> List["Message"]:
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
                piece = MessagePiece(
                    role=prompt.role or "user",
                    original_value=prompt.value,
                    converted_value_data_type=prompt.data_type or "text",
                    prompt_target_identifier={"id": "seed_group_conversion"},
                    conversation_id=str(prompt.prompt_group_id),
                    sequence=prompt.sequence or 0,
                )
                message_pieces.append(piece)
            
            # Create Message from pieces
            messages.append(Message(message_pieces=message_pieces, skip_validation=True))
        
        return messages

    def __repr__(self):
        return f"<SeedGroup(seeds={len(self.prompts)} prompts)>"
