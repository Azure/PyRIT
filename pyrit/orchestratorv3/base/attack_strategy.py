# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import ast
import logging
import uuid
from abc import ABC, abstractmethod
from typing import Generic, Optional

from pyrit.common import default_values
from pyrit.common.logger import logger
from pyrit.memory.central_memory import CentralMemory
from pyrit.models import PromptRequestPiece, Score
from pyrit.models.identifiers import Identifier
from pyrit.orchestratorv3.base.core import BacktrackingStrategy, T_Context, T_Result


class AttackStrategy(ABC, Identifier, Generic[T_Context, T_Result]):
    """
    Base class for attack strategies with enforced lifecycle management

    The lifecycle is enforced by the execute method, which ensures:
    1. setup() is called before the actual attack logic
    2. perform_attack() contains the core attack implementation
    3. teardown() is guaranteed to be called even if exceptions occur
    """

    def __init__(
        self, backtracking_strategy: Optional[BacktrackingStrategy[T_Context]] = None, logger: logging.Logger = logger
    ):
        self._backtracking_strategy = backtracking_strategy
        self._logger = logger
        self._id = uuid.uuid4()
        self._memory = CentralMemory.get_memory_instance()
        self._memory_labels: dict[str, str] = ast.literal_eval(
            default_values.get_non_required_value(env_var_name="GLOBAL_MEMORY_LABELS") or "{}"
        )

    def get_identifier(self):
        return {
            "__type__": self.__class__.__name__,
            "__module__": self.__class__.__module__,
            "id": str(self._id),
        }

    @abstractmethod
    async def _setup(self, *, context: T_Context) -> None:
        """Prepare the attack strategy"""
        pass

    @abstractmethod
    async def _perform_attack(self, *, context: T_Context) -> T_Result:
        """
        Core attack implementation to be defined by subclasses
        This contains the actual attack logic that subclasses must implement
        """
        pass

    @abstractmethod
    async def _teardown(self, *, context: T_Context) -> None:
        """Clean up after attack execution"""
        pass

    async def execute(self, *, context: T_Context) -> T_Result:
        """
        Execute the attack strategy with complete lifecycle management
        This method enforces the setup -> execute -> teardown lifecycle
        """

        result = None
        try:
            # Setup phase
            self._logger.debug("Setting up attack strategy")
            await self._setup(context=context)

            # Execution phase
            self._logger.debug("Executing attack strategy")
            result = await self._perform_attack(context=context)
            return result

        except Exception as e:
            self._logger.exception(f"Error during attack execution: {e}")
            raise
        finally:
            # Teardown phase - guaranteed to run
            self._logger.debug("Tearing down attack strategy")
            await self._teardown(context=context)

    async def check_backtracking(
        self, context: T_Context, response: PromptRequestPiece, score: Optional[Score]
    ) -> bool:
        """Check if backtracking should be applied and apply it if needed"""
        if not self._backtracking_strategy:
            return False

        should_backtrack = await self._backtracking_strategy.should_backtrack(context, response, score)
        if should_backtrack:
            self._logger.info(f"Applying backtracking with strategy {type(self._backtracking_strategy).__name__}")
            await self._backtracking_strategy.apply_backtracking(context)
            return True
        return False
