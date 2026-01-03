# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
SeedAttackGroup - A group of seeds for use in attack scenarios.

Extends SeedGroup to enforce exactly one objective is present.
"""

from __future__ import annotations

from typing import Any, Dict, Sequence, Union

from pyrit.models.seeds.seed import Seed
from pyrit.models.seeds.seed_group import SeedGroup
from pyrit.models.seeds.seed_objective import SeedObjective


class SeedAttackGroup(SeedGroup):
    """
    A group of seeds for use in attack scenarios.

    This class extends SeedGroup with attack-specific validation:
    - Requires exactly one SeedObjective (not optional like in SeedGroup)

    All other functionality (simulated conversation, prepended conversation,
    next_message, etc.) is inherited from SeedGroup.
    """

    def __init__(
        self,
        *,
        seeds: Sequence[Union[Seed, Dict[str, Any]]],
    ):
        """
        Initialize a SeedAttackGroup.

        Args:
            seeds: Sequence of seeds. Must include exactly one SeedObjective.

        Raises:
            ValueError: If seeds is empty.
            ValueError: If exactly one objective is not provided.
        """
        super().__init__(seeds=seeds)
        self._enforce_exactly_one_objective()

    def _enforce_exactly_one_objective(self) -> None:
        """Ensure exactly one objective is present."""
        objective_count = len([s for s in self.seeds if isinstance(s, SeedObjective)])
        if objective_count != 1:
            raise ValueError(
                f"SeedAttackGroup must have exactly one objective. Found {objective_count}."
            )
