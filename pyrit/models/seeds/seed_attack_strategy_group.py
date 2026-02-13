# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
SeedAttackStrategyGroup - A group of seeds representing a general attack strategy.
For example, this includes jailbreaks, roleplays, or other reusable strategies that 
can be applied to multiple objectives.

Extends SeedGroup to enforce that all seeds have is_general_strategy=True.
"""

from __future__ import annotations

from typing import Any, Dict, Sequence, Union

from pyrit.models.seeds.seed import Seed
from pyrit.models.seeds.seed_group import SeedGroup


class SeedAttackStrategyGroup(SeedGroup):
    """
    A group of seeds representing a general attack strategy.

    This class extends SeedGroup with strategy-specific validation:
    - Requires all seeds to have is_general_strategy=True

    All other functionality (simulated conversation, prepended conversation,
    next_message, etc.) is inherited from SeedGroup.
    """

    def __init__(
        self,
        *,
        seeds: Sequence[Union[Seed, Dict[str, Any]]],
    ):
        """
        Initialize a SeedAttackStrategyGroup.

        Args:
            seeds: Sequence of seeds. All seeds must have is_general_strategy=True.

        Raises:
            ValueError: If seeds is empty.
            ValueError: If any seed does not have is_general_strategy=True.
        """
        super().__init__(seeds=seeds)

    def validate(self) -> None:
        """
        Validate the seed attack strategy group state.

        Extends SeedGroup validation to require all seeds to be general strategies.

        Raises:
            ValueError: If validation fails.
        """
        super().validate()
        self._enforce_all_general_strategy()

    def _enforce_all_general_strategy(self) -> None:
        """
        Ensure all seeds have is_general_strategy=True.

        Raises:
            ValueError: If any seed does not have is_general_strategy=True.
        """
        non_general = [seed for seed in self.seeds if not seed.is_general_strategy]
        if non_general:
            non_general_types = [type(s).__name__ for s in non_general]
            raise ValueError(
                f"All seeds in SeedAttackStrategyGroup must have is_general_strategy=True. "
                f"Found {len(non_general)} seed(s) without it: {non_general_types}"
            )
