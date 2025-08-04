# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import ast
import copy
import logging
import time
import uuid
from abc import ABC, abstractmethod
from contextlib import asynccontextmanager
from typing import (
    Any,
    AsyncIterator,
    Dict,
    Generic,
    List,
    MutableMapping,
    Optional,
    overload,
)

from pyrit.attacks.base.attack_context import ContextT
from pyrit.common import default_values, get_kwarg_param
from pyrit.common.logger import logger
from pyrit.exceptions.exception_classes import (
    AttackExecutionException,
    AttackValidationException,
)
from pyrit.memory.central_memory import CentralMemory
from pyrit.models import AttackOutcome, AttackResultT, Identifier, PromptRequestResponse


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
            strategy_info = f"[{strategy_name} (ID: {str(strategy_id)})]"
            return f"{strategy_info} {msg}", kwargs

        return msg, kwargs


class AttackStrategy(ABC, Identifier, Generic[ContextT, AttackResultT]):
    """
    Abstract base class for attack strategies with enforced lifecycle management.

    Ensures a consistent execution flow: validate -> setup -> attack -> teardown.
    The teardown phase is guaranteed to run even if exceptions occur.

    Subclasses must implement:
    _validate_context(): Validate attack context
    _setup_async(): Initialize resources
    _perform_attack_async(): Execute the attack logic
    _teardown_async(): Clean up resources
    """

    def __init__(self, *, context_type: type[ContextT], logger: logging.Logger = logger):
        self._id = uuid.uuid4()
        self._memory = CentralMemory.get_memory_instance()
        self._memory_labels: Dict[str, str] = ast.literal_eval(
            default_values.get_non_required_value(env_var_name="GLOBAL_MEMORY_LABELS") or "{}"
        )

        self._context_type = context_type

        self._logger = AttackStrategyLogAdapter(
            logger, {"strategy_name": self.__class__.__name__, "strategy_id": str(self._id)[:8]}
        )

    def get_identifier(self):
        return {
            "__type__": self.__class__.__name__,
            "__module__": self.__class__.__module__,
            "id": str(self._id),
        }

    def _warn_if_set(self, *, config: Any, unused_fields: List[str]) -> None:
        """
        Utility method to warn about unused parameters in attack configurations.

        This method checks if specified fields in a configuration object are set
        (not None and not empty for collections) and logs a warning message for each
        field that will be ignored by the current attack strategy.

        Args:
            config (Any): The configuration object to check for unused fields.
            unused_fields (List[str]): List of field names to check in the config object.
        """
        config_name = config.__class__.__name__

        for field_name in unused_fields:
            # Get the field value from the config object
            if not hasattr(config, field_name):
                self._logger.warning(
                    f"Field '{field_name}' does not exist in {config_name}. " f"Skipping unused parameter check."
                )
                continue

            param_value = getattr(config, field_name)

            # Check if the parameter is set
            is_set = False
            if param_value is not None:
                # For collections, also check if they are not empty
                if hasattr(param_value, "__len__"):
                    is_set = len(param_value) > 0
                else:
                    is_set = True

            if is_set:
                self._logger.warning(
                    f"{field_name} was provided in {config_name} but is not used by {self.__class__.__name__}. "
                    f"This parameter will be ignored."
                )

    @abstractmethod
    def _validate_context(self, *, context: ContextT) -> None:
        """
        Validate the attack context before execution.
        This method should be implemented by subclasses to validate that the context
        is suitable for the attack strategy.

        Args:
            context (ContextT): The context to validate.

        Raises:
            AttackValidationException: If the context is invalid for this attack strategy.
        """
        pass

    @abstractmethod
    async def _setup_async(self, *, context: ContextT) -> None:
        """
        Setup phase before executing the attack.
        This method should be implemented by subclasses to prepare any necessary state or resources.
        This method is guaranteed to be called before the attack execution and after context validation.

        Args:
            context (ContextT): The context for the attack.
        """
        pass

    @abstractmethod
    async def _perform_attack_async(self, *, context: ContextT) -> AttackResultT:
        """
        Core attack implementation to be defined by subclasses.
        This contains the actual attack logic that subclasses must implement.

        Args:
            context (ContextT): The context for the attack.

        Returns:
            AttackResultT: The result of the attack execution.
        """
        pass

    @abstractmethod
    async def _teardown_async(self, *, context: ContextT) -> None:
        """
        Teardown phase after executing the attack.
        This method should be implemented by subclasses to clean up any resources or state.
        This method is guaranteed to be called even if exceptions occur during execution.

        Args:
            context (ContextT): The context for the attack.
        """
        pass

    def _log_attack_outcome(self, result: AttackResultT) -> None:
        """
        Log the outcome of the attack.

        Args:
            result (AttackResultT): The result of the attack containing outcome and reason.
        """
        attack_name = self.__class__.__name__

        if result.outcome == AttackOutcome.SUCCESS:
            self._logger.info(
                f"{attack_name} achieved the objective. " f"Reason: {result.outcome_reason or 'Not specified'}"
            )
        elif result.outcome == AttackOutcome.UNDETERMINED:
            self._logger.info(
                f"{attack_name} outcome is undetermined. " f"Reason: {result.outcome_reason or 'Not specified'}"
            )
        else:
            self._logger.info(
                f"{attack_name} did not achieve the objective. " f"Reason: {result.outcome_reason or 'Not specified'}"
            )

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
        """
        self._logger.debug(f"Setting up attack strategy for objective: '{context.objective}'")
        try:
            await self._setup_async(context=context)
            yield
        finally:
            self._logger.debug(f"Tearing down attack strategy for objective: '{context.objective}'")
            await self._teardown_async(context=context)

    async def execute_with_context_async(self, *, context: ContextT) -> AttackResultT:
        """
        Execute attack with complete lifecycle management.

        Enforces: validate -> setup -> execute -> teardown.

        Args:
            context (ContextT): The context for the attack, containing configuration and state.

        Returns:
            AttackResultT: The result of the attack execution, including outcome and reason.

        Raises:
            AttackValidationError: If context validation fails.
            AttackExecutionError: If attack execution fails.
        """
        # Validation phase
        # This is a critical step to ensure the context is suitable for the attack
        self._logger.debug("Validating context before attack execution")
        try:
            self._validate_context(context=context)
        except Exception as e:
            error = AttackValidationException(
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
            # Track execution start time
            start_time = time.perf_counter()

            async with self._execution_context(context):
                self._logger.debug(f"Performing attack: {self.__class__.__name__}")
                result = await self._perform_attack_async(context=context)

                # Calculate execution time in milliseconds
                end_time = time.perf_counter()
                execution_time_ms = int((end_time - start_time) * 1000)

                # Set the execution time on the result
                result.execution_time_ms = execution_time_ms

                self._logger.debug(f"Attack execution completed in {execution_time_ms}ms")

                # Log the attack outcome
                self._log_attack_outcome(result)

                # Add the result to memory
                self._memory.add_attack_results_to_memory(attack_results=[result])

                return result

        except (AttackExecutionException, AttackValidationException):
            raise  # Re-raise
        except Exception as e:
            # Create proper execution error with attack details
            objective = context.objective
            exec_error = AttackExecutionException(
                message=f"Unexpected error during attack execution: {str(e)}",
                attack_name=self.__class__.__name__,
                objective=objective,
            )
            exec_error.process_exception()
            raise exec_error from e

    async def execute_async(
        self,
        **kwargs,
    ) -> AttackResultT:
        """
        Execute the attack strategy asynchronously with provided parameters.

        This method creates the appropriate context from the provided parameters and
        executes the attack using the enforced lifecycle (validate -> setup -> execute -> teardown).

        All attack strategies must provide an objective, either:
        - Directly from user input (for strategies that require explicit objectives)
        - Generated from other parameters (for strategies that auto-generate objectives)

        Args:
            **kwargs: Attack parameters. Common parameters include:
                - objective (str): Required. The objective of the attack. This must be provided
                    either directly by the user or generated by subclass logic before calling
                    this method.
                - prepended_conversation (Optional[List[PromptRequestResponse]]): Optional.
                    Conversation to prepend to the target model.
                - memory_labels (Optional[Dict[str, str]]): Optional. Additional labels
                    that can be applied to the prompts throughout the attack.
                - Additional strategy-specific parameters as defined by subclasses.

        Returns:
            AttackResultT: The result of the attack execution, including outcome,
                reason, and execution metadata.

        Raises:
            ValueError: If required parameters (including objective) are missing or invalid.
            TypeError: If parameters are of incorrect types.
            AttackValidationException: If context validation fails.
            AttackExecutionException: If attack execution fails.

        Example:
            ```python
            # For strategies that require explicit objective
            result = await strategy.execute_async(
                objective="Test prompt injection vulnerability",
                prepended_conversation=[previous_interaction],
                memory_labels={"experiment": "baseline_test"}
            )

            # For strategies that generate objectives (handled by subclass)
            # The subclass will generate the objective before calling super().execute_async()
            result = await strategy.execute_async(
                target_model="gpt-4",
                attack_type="jailbreak",
                memory_labels={"experiment": "auto_objective_test"}
            )
            ```
        """
        # Extract common optional parameters
        pc = get_kwarg_param(
            kwargs=kwargs, param_name="prepended_conversation", expected_type=list, required=False, default_value=[]
        )

        # Deep copy the prepended conversation to avoid modifying the original
        prepended_conversation = copy.deepcopy(pc) if pc else []

        # Extract memory_labels if provided, default to empty dict
        memory_labels = (
            get_kwarg_param(
                kwargs=kwargs, param_name="memory_labels", expected_type=dict, required=False, default_value={}
            )
            or {}
        )

        objective = (
            get_kwarg_param(
                kwargs=kwargs,
                param_name="objective",
                expected_type=str,
                required=False,
            )
            or ""
        )

        # Create context using the context type's factory method
        # The context type will handle validation of required parameters
        context = self._context_type.create_from_params(
            objective=objective, prepended_conversation=prepended_conversation, memory_labels=memory_labels, **kwargs
        )

        return await self.execute_with_context_async(context=context)


class AttackStrategyWithObjective(AttackStrategy[ContextT, AttackResultT]):
    """
    Abstract base class for attack strategies that require an explicit objective parameter from users.

    This class provides type-safe overloads that make the objective parameter explicit,
    improving the developer experience with better IDE support and type checking.

    Use this base class for attack strategies where:
    - The objective must be explicitly provided by the user
    - The objective cannot be auto-generated from other parameters
    """

    def __init__(self, *, context_type: type[ContextT], logger: logging.Logger = logger):
        super().__init__(context_type=context_type, logger=logger)

    @overload
    async def execute_async(
        self,
        *,
        objective: str,
        prepended_conversation: Optional[list[PromptRequestResponse]] = None,
        memory_labels: Optional[dict[str, str]] = None,
        **kwargs,
    ) -> AttackResultT:
        """
        Execute the attack strategy asynchronously with the provided parameters.

        Args:
            objective (str): The objective of the attack.
            prepended_conversation (Optional[List[PromptRequestResponse]]): Conversation to prepend.
            memory_labels (Optional[Dict[str, str]]): Memory labels for the attack context.
            **kwargs: Additional parameters for the attack.

        Returns:
            AttackResultT: The result of the attack execution.
        """
        ...

    @overload
    async def execute_async(
        self,
        **kwargs,
    ) -> AttackResultT: ...

    async def execute_async(
        self,
        **kwargs,
    ) -> AttackResultT:
        """
        Execute the attack strategy asynchronously with provided parameters.

        This implementation enforces that an objective parameter is required,
        providing clear error messages if it's missing.

        Args:
            **kwargs: Attack parameters including:
                - objective (str): Required. The objective of the attack.
                - prepended_conversation (Optional[List[PromptRequestResponse]]): Optional.
                    Conversation to prepend to the target model.
                - memory_labels (Optional[Dict[str, str]]): Optional. Additional labels
                    that can be applied to the prompts throughout the attack.
                - Additional strategy-specific parameters as defined by subclasses.

        Returns:
            AttackResultT: The result of the attack execution, including outcome,
                reason, and execution metadata.
        """
        # Extract objective - required at base level since all strategies need one
        # Some get it from user, others generate it, but all must provide it to context
        objective = (
            get_kwarg_param(
                kwargs=kwargs,
                param_name="objective",
                expected_type=str,
            )
            or ""
        )
        return await super().execute_async(objective=objective, **kwargs)
