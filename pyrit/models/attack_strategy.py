# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Union

from pyrit.common.apply_parameters_to_template_to_template import apply_parameters_to_template
from pyrit.common.yaml_loadable import YamlLoadable


@dataclass
class AttackStrategy(YamlLoadable):
    template: str
    parameters: List[str]
    kwargs: Dict[str, str]

    def __init__(self, *, strategy: Union[Path | str], **kwargs):
        self.kwargs = kwargs
        if isinstance(strategy, Path):
            strategy_data = YamlLoadable.from_yaml_file(strategy)
            self.template = strategy_data.template
            self.parameters = strategy_data.parameters
        else:
            self.template = strategy
            self.parameters = list(kwargs.keys())

    def __str__(self):
        """Returns a string representation of the attack strategy."""
        return apply_parameters_to_template(**self.kwargs)
