from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Union

from pyrit.models import PromptTemplate


@dataclass
class ScorerStrategy:
    def __init__(self, *, strategy: Union[Path | str], **kwargs):
        self.kwargs = kwargs
        if isinstance(strategy, Path):
            self.strategy = PromptTemplate.from_yaml_file(strategy)
        else:
            self.strategy = PromptTemplate(template=strategy, parameters=list(kwargs.keys()))

    def __str__(self):
        """Returns a string representation of the scorer strategy."""
        return self.strategy.apply_custom_metaprompt_parameters(**self.kwargs)
