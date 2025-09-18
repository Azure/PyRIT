# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

from abc import ABC
from copy import deepcopy
from dataclasses import dataclass
from typing import Optional, TypeVar

from pyrit.executor.core import StrategyContext

StrategyResultT = TypeVar("StrategyResultT", bound="StrategyResult")


@dataclass
class StrategyResult(ABC):
    """
    Base class for all strategy results.
    The context attribute is used by strategies which return a 
    partial result and need to be called again with the same context.
    """

    def duplicate(self: StrategyResultT) -> StrategyResultT:
        """
        Create a deep copy of the result.

        Returns:
            StrategyResult: A deep copy of the result.
        """
        return deepcopy(self)

@dataclass
class StrategyResultIntermediate(StrategyResult):
    """
    Decorator for StrategyResult to indicate that the result is intermediate
    and that the strategy should be invoked again with the same context.
    """
    context: Optional[StrategyContext] = None
    
    @property.getter
    def context(self) -> Optional[StrategyContext]:
        return self._context
    
    @property.setter
    def context(self, value: Optional[StrategyContext]) -> None:
        self._context = value
    
    @property.getter
    def final(self) -> bool:
        return self.context is None
    
    @property.setter
    def final(self, value: bool) -> None:
        if value:
            self.context = None
        else:
            raise ValueError("Cannot set final to False; provide a valid context instead.")