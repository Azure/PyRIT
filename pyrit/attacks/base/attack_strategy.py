# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import ast
import logging
import uuid
from abc import ABC, abstractmethod
from typing import Generic, Optional

from pyrit.attacks.base.backtracking_strategy import BacktrackingStrategy
from pyrit.attacks.base.context import ContextT
from pyrit.attacks.base.result import ResultT
from pyrit.common import default_values
from pyrit.common.logger import logger
from pyrit.memory.central_memory import CentralMemory
from pyrit.models.identifiers import Identifier


class AttackStrategy(ABC, Identifier, Generic[ContextT, ResultT]):
    """
    Base class for attack strategies with enforced lifecycle management

    The lifecycle is enforced by the execute method, which ensures:
    1. setup() is called before the actual attack logic
    2. perform_attack() contains the core attack implementation
    3. teardown() is guaranteed to be called even if exceptions occur
    """

    def __init__(
        self, backtracking_strategy: Optional[BacktrackingStrategy[ContextT]] = None, logger: logging.Logger = logger
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
    def _validate_context(self, *, context: ContextT) -> None:
        """
        Validate the context before executing the attack
        This method should be implemented by subclasses to ensure the context is suitable for the attack

        Args:
            context (ContextT): The context to validate
        """
        pass

    @abstractmethod
    async def _setup_async(self, *, context: ContextT) -> None:
        """
        Setup phase before executing the attack
        This method should be implemented by subclasses to prepare any necessary state or resources
        This method is called before the actual attack logic in execute()

        Args:
            context (ContextT): The context for the attack
        """
        pass

    @abstractmethod
    async def _perform_attack_async(self, *, context: ContextT) -> ResultT:
        """
        Core attack implementation to be defined by subclasses
        This contains the actual attack logic that subclasses must implement

        Args:
            context (ContextT): The context for the attack

        Returns:
            ResultT: The result of the attack execution
        """
        pass

    @abstractmethod
    async def _teardown_async(self, *, context: ContextT) -> None:
        """
        Teardown phase after executing the attack
        This method should be implemented by subclasses to clean up any resources or state
        This method is guaranteed to be called even if exceptions occur during execution

        Args:
            context (ContextT): The context for the attack
        """
        pass

    async def execute_async(self, *, context: ContextT) -> ResultT:
        """
        Execute the attack strategy with complete lifecycle management
        This method enforces the validate -> setup -> execute -> teardown lifecycle

        Args:
            context (ContextT): The context for the attack

        Returns:
            ResultT: The result of the attack execution
        """

        # Validating context before proceeding
        self._logger.debug("Validating context before attack execution")
        self._validate_context(context=context)

        result = None
        try:
            # Setup phase
            self._logger.debug("Setting up attack strategy")
            await self._setup_async(context=context)

            # Execution phase
            self._logger.debug("Executing attack strategy")
            result = await self._perform_attack_async(context=context)
            return result

        except Exception as e:
            self._logger.exception(f"Error during attack execution: {e}")
            raise
        finally:
            # Teardown phase - guaranteed to run
            self._logger.debug("Tearing down attack strategy")
            await self._teardown_async(context=context)
