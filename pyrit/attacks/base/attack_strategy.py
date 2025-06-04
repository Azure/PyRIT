# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import ast
import logging
import uuid
from abc import ABC, abstractmethod
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Generic, MutableMapping

from pyrit.attacks.base.context import ContextT
from pyrit.attacks.base.result import ResultT
from pyrit.common import default_values
from pyrit.common.logger import logger
from pyrit.exceptions.exception_classes import (
    AttackExecutionError,
    AttackValidationError,
)
from pyrit.memory.central_memory import CentralMemory
from pyrit.models.identifiers import Identifier


class AttackStrategyLogAdapter(logging.LoggerAdapter):
    """
    Custom logger adapter that adds attack strategy information to log messages.
    """

    def process(self, msg: Any, kwargs: MutableMapping[str, Any]) -> tuple[str, MutableMapping[str, Any]]:
        """Add attack strategy context to each log message."""
        if not self.extra:
            return msg, kwargs

        # Extract strategy information from extra
        strategy_name = self.extra.get("strategy_name")
        strategy_id = self.extra.get("strategy_id")

        if strategy_name and strategy_id:
            strategy_info = f"[{strategy_name} (ID: {strategy_id})]"
            return f"{strategy_info} {msg}", kwargs

        return msg, kwargs


class AttackStrategy(ABC, Identifier, Generic[ContextT, ResultT]):
    """
    Base class for attack strategies with enforced lifecycle management

    The lifecycle is enforced by the execute method, which ensures:
    1. setup() is called before the actual attack logic
    2. perform_attack() contains the core attack implementation
    3. teardown() is guaranteed to be called even if exceptions occur
    """

    def __init__(self, *, logger: logging.Logger = logger):
        self._id = uuid.uuid4()
        self._memory = CentralMemory.get_memory_instance()
        self._memory_labels: dict[str, str] = ast.literal_eval(
            default_values.get_non_required_value(env_var_name="GLOBAL_MEMORY_LABELS") or "{}"
        )

        self._logger = AttackStrategyLogAdapter(
            logger, {"strategy_name": self.__class__.__name__, "strategy_id": str(self._id)[:8]}
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

    @asynccontextmanager
    async def _execution_context(self, context: ContextT) -> AsyncIterator[None]:
        """
        Manages the complete lifecycle of an attack execution as an async context manager.

        This method provides a context manager that ensures proper setup and teardown
        of attack resources, regardless of whether the attack completes successfully
        or encounters an error.

        Args:
            context: The execution context containing configuration and state for the attack.

        Yields:
            None: Control is yielded back to the caller after setup is complete.

        Note:
            - Setup is performed before yielding control
            - Teardown is guaranteed to run even if an exception occurs
            - All setup and teardown operations are logged for debugging
        """
        self._logger.debug(f"Setting up attack strategy for objective: '{context.objective}'")
        try:
            await self._setup_async(context=context)
            yield
        finally:
            self._logger.debug(f"Tearing down attack strategy for objective: '{context.objective}'")
            await self._teardown_async(context=context)

    async def execute_async(self, *, context: ContextT) -> ResultT:
        """
        Execute attack with complete lifecycle management.

        Enforces: validate -> setup -> execute -> teardown

        Args:
            context: Attack execution context

        Returns:
            Result of the attack execution

        Raises:
            AttackValidationError: If context validation fails
            AttackExecutionError: If attack execution fails
        """
        # Validation phase
        # This is a critical step to ensure the context is suitable for the attack
        self._logger.debug("Validating context before attack execution")
        try:
            self._validate_context(context=context)
        except Exception as e:
            error = AttackValidationError(
                message=f"Context validation failed: {str(e)}",
                context_info={
                    "attack_type": self.__class__.__name__,
                    "original_error": str(e),
                    "error_type": type(e).__name__,
                },
            )
            error.process_exception()
            raise error from e

        # Execution with lifecycle management
        # This uses an async context manager to ensure setup and teardown are handled correctly
        try:
            async with self._execution_context(context):
                self._logger.debug(f"Performing attack: {self.__class__.__name__}")
                return await self._perform_attack_async(context=context)
        except (AttackExecutionError, AttackValidationError):
            raise  # Re-raise
        except Exception as e:
            # Create proper execution error with attack details
            objective = context.objective
            exec_error = AttackExecutionError(
                message=f"Unexpected error during attack execution: {str(e)}",
                attack_name=self.__class__.__name__,
                objective=objective,
            )
            exec_error.process_exception()
            raise exec_error from e
