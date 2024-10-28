# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Union

from pyrit.common.apply_parameters_to_template import apply_parameters_to_template


@dataclass
class AttackStrategy:
    template: str
    parameters: List[str]
    kwargs: Dict[str, str]

    def __init__(self, *, strategy: Union[Path | str], **kwargs):
        self.kwargs = kwargs
        if isinstance(strategy, Path):
            from pyrit.models import SeedPromptTemplate
            strategy_data = SeedPromptTemplate.from_yaml_file(strategy)
            self.template = strategy_data.value
            self.parameters = strategy_data.parameters
        else:
            self.template = strategy
            self.parameters = list(kwargs.keys())

    def __str__(self):
        """Returns a string representation of the attack strategy."""
        return apply_parameters_to_template(template=self.template, parameters=self.parameters, **self.kwargs)
