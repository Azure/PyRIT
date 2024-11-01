# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Union


@dataclass
class AttackStrategy:
    """
    This is deprecated and will be removed soon.
    Probably the next PR. But waiting to avoid conflicts.
    """

    value: str
    parameters: List[str]
    kwargs: Dict[str, str]

    def __init__(self, *, strategy: Union[Path | str], **kwargs):
        from pyrit.models import SeedPrompt

        self.kwargs = kwargs
        if isinstance(strategy, Path):
            self.seedprompt = SeedPrompt.from_yaml_file(strategy)
        else:
            self.seedprompt = SeedPrompt(value=strategy, data_type="text", parameters=list(kwargs.keys()))

    def __str__(self):
        """Returns a string representation of the attack strategy."""
        return self.seedprompt.render(**self.kwargs)
