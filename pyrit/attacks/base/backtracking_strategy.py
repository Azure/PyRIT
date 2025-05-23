# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from abc import ABC, abstractmethod
from typing import Generic, Optional

from pyrit.attacks.base.context import ContextT
from pyrit.models.prompt_request_piece import PromptRequestPiece
from pyrit.models.score import Score


class BacktrackingStrategy(ABC, Generic[ContextT]):
    """Base class for backtracking strategies"""

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the name of the backtracking strategy"""
        pass

    @abstractmethod
    async def should_backtrack(self, context: ContextT, response: PromptRequestPiece, score: Optional[Score]) -> bool:
        """Determine if backtracking should be applied"""
        pass

    @abstractmethod
    async def apply_backtracking(self, context: ContextT) -> None:
        """Apply backtracking to the context"""
        pass
