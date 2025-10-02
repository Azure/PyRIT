# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

from abc import ABC
from dataclasses import dataclass
from typing import Optional, TypeVar

from pyrit.common import Duplicable
from pyrit.executor.core import StrategyContext

StrategyResultT = TypeVar("StrategyResultT", bound="StrategyResult")

@dataclass
class StrategyResult(ABC, Duplicable):
    """
    Base class for all strategy results.
    
    Life cycle of an interactive attack:
    Context t = 0 -> Result.context not None (Result NOT final)
    Context t = 1 -> Result.context not None (Result NOT final)
    ...
    Context t = n -> Result.context is None (Result IS final)
    
    Design decision: one attribute for context; we assume
    no preservation of context for the final result since 
    context is designed to be ephemeral and belongs to
    Strategy and its async lifecycle methods.
    
    Attributes:
        context (Optional[StrategyContext]): The context associated with this result.
        is None when result is final; the caller is responsible for finalizing the
        StrategyResult, and nothing calling StrategyResult should be modifying its fields.
    """
    
    @property
    def context(self) -> Optional[StrategyContext]:
        """
        If the caller is not interactive, this should be None.
        If it is, then the context attribute is the context that should be passed
        to the next round of the strategy.
        """
        return self._context
    
    @property.setter
    def context(self, value: Optional[StrategyContext]):
        """
        Each timestep produces a new result object and a new context object
        to prevent passing by reference; otherwise one strategy could modify
        one or more context objects, and one context object could be used in
        one or more results.
        """
        self._context = value.duplicate()