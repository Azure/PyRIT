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
    
    context: Optional[StrategyContext] = None

    def duplicate(self: StrategyResultT) -> StrategyResultT:
        """
        Create a deep copy of the result.

        Returns:
            StrategyResult: A deep copy of the result.
        """
        return deepcopy(self)
