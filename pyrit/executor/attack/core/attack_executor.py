# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import asyncio
from dataclasses import dataclass
from typing import Any, Dict, Generic, List, Optional, Sequence, TypeVar, Union, cast

from pyrit.common.logger import logger
from pyrit.executor.attack.core import (
    AttackStrategy,
    AttackStrategyContextT,
    AttackStrategyResultT,
)
from pyrit.executor.attack.multi_turn.multi_turn_attack_strategy import (
    MultiTurnAttackContext,
)
from pyrit.executor.attack.single_turn.single_turn_attack_strategy import (
    SingleTurnAttackContext,
)
from pyrit.models import Message, SeedGroup

AttackResultT = TypeVar("AttackResultT")


@dataclass
class AttackExecutorResult(Generic[AttackResultT]):
    """
    Result container for attack execution, supporting both full and partial completion.

    This class holds results from parallel attack execution. It is iterable and
    behaves like a list in the common case where all objectives complete successfully.

    When some objectives don't complete (throw exceptions), access incomplete_objectives
    to retrieve the failures, or use raise_if_incomplete() to raise the first exception.

    Note: "completed" means the execution finished, not that the attack objective was achieved.
    """

    completed_results: List[AttackResultT]
    incomplete_objectives: List[tuple[str, BaseException]]

    def __iter__(self):
        """
        Iterate over completed results.

        Returns:
            Iterator[AttackResultT]: An iterator over the completed results.
        """
        return iter(self.completed_results)

    def __len__(self) -> int:
        """
        Return number of completed results.

        Returns:
            int: The number of completed results.
        """
        return len(self.completed_results)

    def __getitem__(self, index: int) -> AttackResultT:
        """
        Access completed results by index.

        Returns:
            AttackResultT: The completed result at the specified index.
        """
        return self.completed_results[index]

    @property
    def has_incomplete(self) -> bool:
        """Check if any objectives didn't complete execution."""
        return len(self.incomplete_objectives) > 0

    @property
    def all_completed(self) -> bool:
        """Check if all objectives completed execution."""
        return len(self.incomplete_objectives) == 0

    @property
    def exceptions(self) -> List[BaseException]:
        """
        Get all exceptions from incomplete objectives.

        Returns:
            List[BaseException]: List of exceptions that caused objectives to fail.
        """
        return [exception for _, exception in self.incomplete_objectives]

    def raise_if_incomplete(self) -> None:
        """
        Raise the first exception if any objectives are incomplete.

        Raises:
            BaseException: The first exception from incomplete objectives.
        """
        if self.incomplete_objectives:
            raise self.incomplete_objectives[0][1]

    def get_results(self) -> List[AttackResultT]:
        """
        Get completed results, raising if any incomplete.

        Returns:
            List[AttackResultT]: All completed results.

        Raises:
            BaseException: The first exception from incomplete objectives.
        """
        self.raise_if_incomplete()
        return self.completed_results


class AttackExecutor:
    """
    Manages the execution of attack strategies with support for different execution patterns.

    The AttackExecutor provides controlled execution of attack strategies with features like
    concurrency limiting and parallel execution. It can handle multiple objectives against
    the same target or execute different strategies concurrently.
    """

    _SingleTurnContextT = TypeVar("_SingleTurnContextT", bound=SingleTurnAttackContext)
    _MultiTurnContextT = TypeVar("_MultiTurnContextT", bound=MultiTurnAttackContext)

    def __init__(self, *, max_concurrency: int = 1):
        """
        Initialize the attack executor with configurable concurrency control.

        Args:
            max_concurrency (int): Maximum number of concurrent attack executions allowed.
                Must be a positive integer (defaults to 1).

        Raises:
            ValueError: If max_concurrency is not a positive integer.
        """
        if max_concurrency <= 0:
            raise ValueError(f"max_concurrency must be a positive integer, got {max_concurrency}")
        self._max_concurrency = max_concurrency

    def _filter_params_for_attack(
        self,
        *,
        attack: AttackStrategy[AttackStrategyContextT, AttackStrategyResultT],
        params: Dict[str, Any],
        strict_param_matching: bool,
    ) -> Dict[str, Any]:
        """
        Filter parameters to only those accepted by the attack's context type.

        Uses the attack strategy's accepted_context_parameters property to determine
        what parameters are accepted. This property already excludes parameters that
        the strategy explicitly rejects via _excluded_context_parameters.

        Args:
            attack: The attack strategy to check parameters against.
            params: The parameters to filter.
            strict_param_matching: If True, raise ValueError for unsupported parameters.

        Returns:
            Dict[str, Any]: Filtered parameters.

        Raises:
            ValueError: If strict_param_matching is True and unsupported parameters are provided.
        """
        accepted = attack.accepted_context_parameters

        filtered = {}
        unsupported = []

        for key, value in params.items():
            if value is not None:
                if key in accepted:
                    filtered[key] = value
                else:
                    unsupported.append(key)

        if unsupported and strict_param_matching:
            raise ValueError(
                f"Attack {attack.__class__.__name__} does not accept parameters: {unsupported}. "
                f"Accepted parameters: {accepted}"
            )

        return filtered

    async def execute_attack_async(
        self,
        *,
        attack: AttackStrategy[AttackStrategyContextT, AttackStrategyResultT],
        objectives: Sequence[str],
        prepended_conversations: Optional[Sequence[Optional[List[Message]]]] = None,
        next_messages: Optional[Sequence[Optional[Message]]] = None,
        memory_labels: Optional[Union[Dict[str, str], Sequence[Optional[Dict[str, str]]]]] = None,
        per_attack_params: Optional[Sequence[Dict[str, Any]]] = None,
        strict_param_matching: bool = False,
        return_partial_on_failure: bool = False,
        **broadcast_attack_params,
    ) -> AttackExecutorResult[AttackStrategyResultT]:
        """
        Run attack.execute_async in parallel for each set of parameters.

        This method accepts lists of parameters for execute_async.
        Each execution receives its own set of parameters (objective, prepended_conversation, next_message,
        memory_labels, and any attack-specific params). Broadcast parameters are applied to all executions,
        while per-execution parameters can be customized via corresponding list indices.

        When an attack doesn't accept a parameter supplied (e.g., not all attacks accept next_message), the
        parameter is silently ignored and the attack executes with the parameters it does accept.
        Set strict_param_matching=True to raise an error for unsupported parameters instead.

        Args:
            attack (AttackStrategy): The attack strategy to execute.
            objectives (List[str]): List of attack objectives to test.
            prepended_conversations (Optional[List[Optional[List[Message]]]]): Conversations to prepend
                for each objective. If provided, must match the length of objectives.
            next_messages (Optional[List[Optional[Message]]]): Messages to send for each objective.
                If provided, must match the length of objectives.
            memory_labels (Optional[Union[Dict[str, str], List[Optional[Dict[str, str]]]]]): Memory labels
                for the attack. Can be a single dict (applied to all objectives) or a list of dicts
                (one per objective). If a list, must match the length of objectives.
            per_attack_params (Optional[List[Dict[str, Any]]]): Per-objective attack parameters.
                If provided, must match the length of objectives. Each dict is merged with
                broadcast_attack_params for that objective's execution.
            strict_param_matching (bool): If True, raises ValueError when parameters are provided that
                the attack doesn't accept. If False (default), unsupported parameters are silently ignored.
            return_partial_on_failure (bool): If True, returns partial results when some objectives fail.
                If False (default), raises the first exception encountered.
            **broadcast_attack_params: Additional parameters applied to all objectives.
                These are merged with per-objective per_attack_params (per-objective takes precedence).

        Returns:
            AttackExecutorResult[AttackStrategyResultT]: Result container with completed results and
                any incomplete objectives. The result is iterable and behaves like a list of completed results.

        Raises:
            ValueError: If list parameters don't match the length of objectives, or if
                strict_param_matching is True and unsupported parameters are provided.
            BaseException: If return_partial_on_failure=False and any objective doesn't complete.

        Example:
            >>> executor = AttackExecutor(max_concurrency=3)
            >>> results = await executor.execute_attack_async(
            ...     attack=red_teaming_attack,
            ...     objectives=["objective 1", "objective 2"],
            ...     memory_labels={"scenario": "test"},  # Applied to all
            ...     max_turns=5,  # Broadcast param applied to all
            ... )
            >>> for result in results:
            ...     print(result)
        """
        if not objectives:
            raise ValueError("At least one objective must be provided")

        # Validate list lengths
        if prepended_conversations is not None and len(prepended_conversations) != len(objectives):
            raise ValueError(
                f"Number of prepended_conversations ({len(prepended_conversations)}) must match "
                f"number of objectives ({len(objectives)})"
            )

        if next_messages is not None and len(next_messages) != len(objectives):
            raise ValueError(
                f"Number of next_messages ({len(next_messages)}) must match "
                f"number of objectives ({len(objectives)})"
            )

        if per_attack_params is not None and len(per_attack_params) != len(objectives):
            raise ValueError(
                f"Number of per_attack_params ({len(per_attack_params)}) must match "
                f"number of objectives ({len(objectives)})"
            )

        # Handle memory_labels - can be single dict or list
        memory_labels_list: List[Optional[Dict[str, str]]]
        if memory_labels is None:
            memory_labels_list = [None] * len(objectives)
        elif isinstance(memory_labels, dict):
            # Single dict - broadcast to all objectives
            memory_labels_list = [memory_labels] * len(objectives)
        else:
            # List of dicts
            if len(memory_labels) != len(objectives):
                raise ValueError(
                    f"Number of memory_labels ({len(memory_labels)}) must match "
                    f"number of objectives ({len(objectives)})"
                )
            memory_labels_list = list(memory_labels)

        semaphore = asyncio.Semaphore(self._max_concurrency)

        async def execute_with_semaphore(
            objective: str,
            prepended_conversation: Optional[List[Message]],
            next_message: Optional[Message],
            labels: Optional[Dict[str, str]],
            per_objective_params: Optional[Dict[str, Any]],
        ) -> AttackStrategyResultT:
            async with semaphore:
                # Build params for this execution
                # Start with broadcast params, then merge per-objective params (per-objective takes precedence)
                params = {
                    "objective": objective,
                    "prepended_conversation": prepended_conversation,
                    "next_message": next_message,
                    "memory_labels": labels or {},
                    **broadcast_attack_params,
                }

                # Merge per-objective params (they override broadcast params)
                if per_objective_params:
                    params.update(per_objective_params)

                # Filter params based on what the attack accepts
                filtered_params = self._filter_params_for_attack(
                    attack=attack,
                    params=params,
                    strict_param_matching=strict_param_matching,
                )

                return await attack.execute_async(**filtered_params)

        # Create tasks for each objective
        tasks = []
        for i, objective in enumerate(objectives):
            prepended_conv = prepended_conversations[i] if prepended_conversations else None
            next_msg = next_messages[i] if next_messages else None
            labels = memory_labels_list[i]
            per_obj_params = per_attack_params[i] if per_attack_params else None

            task = execute_with_semaphore(
                objective=objective,
                prepended_conversation=prepended_conv,
                next_message=next_msg,
                labels=labels,
                per_objective_params=per_obj_params,
            )
            tasks.append(task)

        results_or_exceptions = await asyncio.gather(*tasks, return_exceptions=True)

        return self._process_execution_results(
            objectives=objectives,
            results_or_exceptions=results_or_exceptions,
            return_partial_on_failure=return_partial_on_failure,
        )

    async def execute_attack_from_seed_groups_async(
        self,
        *,
        attack: AttackStrategy[AttackStrategyContextT, AttackStrategyResultT],
        seed_groups: Sequence[SeedGroup],
        memory_labels: Optional[Union[Dict[str, str], Sequence[Optional[Dict[str, str]]]]] = None,
        per_attack_params: Optional[Sequence[Dict[str, Any]]] = None,
        strict_param_matching: bool = False,
        return_partial_on_failure: bool = False,
        **broadcast_attack_params,
    ) -> AttackExecutorResult[AttackStrategyResultT]:
        """
        Run attack.execute_async in parallel, extracting parameters from SeedGroups.

        This method extracts attack parameters from SeedGroup objects:
        - objective: From SeedGroup.objective.value (required)
        - prepended_conversation: From SeedGroup.prepended_conversation
        - next_message: From SeedGroup.next_message

        When an attack doesn't accept a parameter supplied (e.g., not all attacks accept next_message), the
        parameter is silently ignored and the attack executes with the parameters it does accept.
        Set strict_param_matching=True to raise an error for unsupported parameters instead.

        Args:
            attack (AttackStrategy): The attack strategy to execute.
            seed_groups (Sequence[SeedGroup]): SeedGroups containing objectives and prompts.
            memory_labels (Optional[Union[Dict[str, str], List[Optional[Dict[str, str]]]]]): Memory labels
                for the attack. Can be a single dict (applied to all) or a list (one per seed group).
            per_attack_params (Optional[List[Dict[str, Any]]]): Per-seed-group attack parameters.
                If provided, must match the length of seed_groups.
            strict_param_matching (bool): If True, raises ValueError when parameters are provided that
                the attack doesn't accept. If False (default), unsupported parameters are silently ignored.
            return_partial_on_failure (bool): If True, returns partial results when some objectives fail.
                If False (default), raises the first exception encountered.
            **broadcast_attack_params: Additional parameters applied to all seed groups.

        Returns:
            AttackExecutorResult[AttackStrategyResultT]: Result container with completed results and
                any incomplete objectives.

        Raises:
            ValueError: If any SeedGroup doesn't have an objective, or if memory_labels list
                doesn't match seed_groups length.
            BaseException: If return_partial_on_failure=False and any objective doesn't complete.

        Example:
            >>> executor = AttackExecutor(max_concurrency=3)
            >>> seed_groups = [SeedGroup(...), SeedGroup(...)]
            >>> results = await executor.execute_attack_from_seed_groups_async(
            ...     attack=prompt_sending_attack,
            ...     seed_groups=seed_groups,
            ... )
        """
        if not seed_groups:
            raise ValueError("At least one seed_group must be provided")

        # Extract parameters from seed groups
        objectives: List[str] = []
        prepended_conversations: List[Optional[List[Message]]] = []
        next_messages: List[Optional[Message]] = []

        for i, seed_group in enumerate(seed_groups):
            if seed_group.objective is None:
                raise ValueError(f"SeedGroup at index {i} does not have an objective")
            objectives.append(seed_group.objective.value)
            prepended_conversations.append(seed_group.prepended_conversation)
            next_messages.append(seed_group.next_message)

        # Build per_attack_params by extracting attack-specific properties from seed groups
        merged_per_attack_params = self._build_per_attack_params_from_seed_groups(
            seed_groups=seed_groups,
            per_attack_params=per_attack_params,
        )

        return await self.execute_attack_async(
            attack=attack,
            objectives=objectives,
            prepended_conversations=prepended_conversations,
            next_messages=next_messages,
            memory_labels=memory_labels,
            per_attack_params=merged_per_attack_params,
            strict_param_matching=strict_param_matching,
            return_partial_on_failure=return_partial_on_failure,
            **broadcast_attack_params,
        )

    def _build_per_attack_params_from_seed_groups(
        self,
        *,
        seed_groups: Sequence[SeedGroup],
        per_attack_params: Optional[Sequence[Dict[str, Any]]] = None,
    ) -> Optional[List[Dict[str, Any]]]:
        """
        Build per-attack parameters by extracting attack-specific properties from seed groups.

        This method extracts properties from SeedGroups that are specific to certain attacks
        (e.g., messages for MultiPromptSendingAttack) and merges them with any user-provided
        per_attack_params. User-provided params take precedence over extracted properties.

        The extracted parameters will be filtered by _filter_params_for_attack later,
        so attacks that don't accept certain parameters will simply ignore them.

        Args:
            seed_groups (Sequence[SeedGroup]): SeedGroups to extract properties from.
            per_attack_params (Optional[Sequence[Dict[str, Any]]]): User-provided per-attack params.

        Returns:
            Optional[List[Dict[str, Any]]]: Merged parameters, or None if no params to add.
        """
        # Check if any seed group has properties we want to extract
        has_messages = any(sg.user_messages for sg in seed_groups)

        # If no special properties and no user params, return None
        if not has_messages and not per_attack_params:
            return None

        merged_params: List[Dict[str, Any]] = []
        for i, seed_group in enumerate(seed_groups):
            # Start with user-provided params if available
            params: Dict[str, Any] = {}
            if per_attack_params and i < len(per_attack_params):
                params = dict(per_attack_params[i])

            # Extract messages (for MultiPromptSendingAttack and similar)
            # Only add if not already specified by user
            if "messages" not in params:
                messages = seed_group.user_messages
                if messages:
                    params["messages"] = messages

            # Add more property extractions here as needed:
            # if "some_property" not in params:
            #     value = seed_group.some_property
            #     if value:
            #         params["some_property"] = value

            merged_params.append(params)

        # If all params are empty, return None to avoid unnecessary processing
        if all(not p for p in merged_params):
            return None

        return merged_params

    async def execute_multi_objective_attack_async(
        self,
        *,
        attack: AttackStrategy[AttackStrategyContextT, AttackStrategyResultT],
        objectives: List[str],
        prepended_conversation: Optional[List[Message]] = None,
        memory_labels: Optional[Dict[str, str]] = None,
        return_partial_on_failure: bool = False,
        **attack_params,
    ) -> AttackExecutorResult[AttackStrategyResultT]:
        """
        Execute the same attack strategy with multiple objectives against the same target in parallel.

        .. deprecated::
            Use :meth:`execute_attack_async` instead. This method will be removed in a future version.

        This method provides a simplified interface for executing multiple objectives without
        requiring users to create context objects. It uses the attack's execute_async method
        which accepts parameters directly.

        Args:
            attack (AttackStrategy[ContextT, AttackStrategyResultT]): The attack strategy to use for all objectives.
            objectives (List[str]): List of attack objectives to test.
            prepended_conversation (Optional[List[Message]]): Conversation to prepend to the target model.
            memory_labels (Optional[Dict[str, str]]): Additional labels that can be applied to the prompts.
            return_partial_on_failure (bool): If True, returns AttackExecutorResult with completed results
                even when some objectives don't complete execution. If False, raises the first exception encountered.
                Defaults to False (raise on failure).
            **attack_params: Additional parameters specific to the attack strategy.

        Returns:
            AttackExecutorResult[AttackStrategyResultT]: Result container with completed results and
                any incomplete objectives. The result is iterable and behaves like a list of completed results.
                Use .has_incomplete or .raise_if_incomplete() to handle failures.

        Raises:
            BaseException: If return_partial_on_failure=False and any objective doesn't complete execution.

        Example:
            >>> executor = AttackExecutor(max_concurrency=3)
            >>> results = await executor.execute_multi_objective_attack_async(
            ...     attack=red_teaming_attack,
            ...     objectives=["how to make a Molotov cocktail", "how to escalate privileges"],
            ... )
            >>> # Iterate directly over results
            >>> for result in results:
            ...     print(result)
        """
        logger.warning(
            "execute_multi_objective_attack_async is deprecated and will disappear in 0.13.0. "
            "Use execute_attack_async instead."
        )

        # Broadcast prepended_conversation to all objectives
        prepended_conversations_list: Optional[List[Optional[List[Message]]]] = cast(
            Optional[List[Optional[List[Message]]]],
            [prepended_conversation] * len(objectives) if prepended_conversation else None,
        )

        return await self.execute_attack_async(
            attack=attack,
            objectives=objectives,
            prepended_conversations=prepended_conversations_list,
            memory_labels=memory_labels,
            return_partial_on_failure=return_partial_on_failure,
            **attack_params,
        )

    async def execute_single_turn_attacks_async(
        self,
        *,
        attack: AttackStrategy[_SingleTurnContextT, AttackStrategyResultT],
        objectives: List[str],
        messages: Optional[List[Message]] = None,
        prepended_conversations: Optional[List[List[Message]]] = None,
        memory_labels: Optional[Dict[str, str]] = None,
        return_partial_on_failure: bool = False,
        **attack_params,
    ) -> AttackExecutorResult[AttackStrategyResultT]:
        """
        Execute a batch of single-turn attacks with multiple objectives.

        .. deprecated::
            Use :meth:`execute_attack_async` instead. This method will be removed in a future version.

        This method is specifically designed for single-turn attacks, allowing you to
        execute multiple objectives in parallel while managing the contexts and prompts.

        Args:
            attack (AttackStrategy[_SingleTurnContextT, AttackStrategyResultT]): The single-turn attack strategy to use,
                the context must be a SingleTurnAttackContext or a subclass of it.
            objectives (List[str]): List of attack objectives to test.
            messages (Optional[List[Message]]): List of messages to use for this execution.
                If provided, must match the length of objectives. Message will be sent along the objective
                with the same list index.
            prepended_conversations (Optional[List[List[Message]]]): Conversations to prepend to each
                objective. If provided, must match the length of objectives. Conversation will be sent along the
                objective with the same list index.
            memory_labels (Optional[Dict[str, str]]): Additional labels that can be applied to the prompts.
                The labels will be the same across all executions.
            return_partial_on_failure (bool): If True, returns AttackExecutorResult with completed results
                even when some objectives don't complete execution. If False, raises the first exception encountered.
                Defaults to False (raise on failure).
            **attack_params: Additional parameters specific to the attack strategy.

        Returns:
            AttackExecutorResult[AttackStrategyResultT]: Result container with completed results and
                any incomplete objectives. The result is iterable and behaves like a list of completed results.

        Raises:
            BaseException: If return_partial_on_failure=False and any objective doesn't complete execution.
            TypeError: If the attack strategy does not use SingleTurnAttackContext.

        Example:
            >>> executor = AttackExecutor(max_concurrency=3)
            >>> results = await executor.execute_single_turn_attacks_async(
            ...     attack=single_turn_attack,
            ...     objectives=["how to make a Molotov cocktail", "how to escalate privileges"],
            ...     messages=[Message(...), Message(...)]
            ... )
        """
        logger.warning(
            "execute_single_turn_attacks_async is deprecated and will disappear in 0.13.0. "
            "Use execute_attack_async instead."
        )

        # Validate that the attack uses SingleTurnAttackContext
        if hasattr(attack, "_context_type") and not issubclass(attack._context_type, SingleTurnAttackContext):
            raise TypeError(
                f"Attack strategy {attack.__class__.__name__} must use SingleTurnAttackContext or a subclass of it."
            )

        return await self.execute_attack_async(
            attack=attack,
            objectives=objectives,
            prepended_conversations=cast(Optional[List[Optional[List[Message]]]], prepended_conversations),
            next_messages=cast(Optional[List[Optional[Message]]], messages),
            memory_labels=memory_labels,
            return_partial_on_failure=return_partial_on_failure,
            **attack_params,
        )

    async def execute_multi_turn_attacks_async(
        self,
        *,
        attack: AttackStrategy[_MultiTurnContextT, AttackStrategyResultT],
        objectives: List[str],
        messages: Optional[List[Message]] = None,
        prepended_conversations: Optional[List[List[Message]]] = None,
        memory_labels: Optional[Dict[str, str]] = None,
        return_partial_on_failure: bool = False,
        **attack_params,
    ) -> AttackExecutorResult[AttackStrategyResultT]:
        """
        Execute a batch of multi-turn attacks with multiple objectives.

        .. deprecated::
            Use :meth:`execute_attack_async` instead. This method will be removed in a future version.

        This method is specifically designed for multi-turn attacks, allowing you to
        execute multiple objectives in parallel while managing the contexts and messages.

        Args:
            attack (AttackStrategy[_MultiTurnContextT, AttackStrategyResultT]): The multi-turn attack strategy to use,
                the context must be a MultiTurnAttackContext or a subclass of it.
            objectives (List[str]): List of attack objectives to test.
            messages (Optional[List[Message]]): List of messages to use for this execution.
                If provided, must match the length of objectives. Messages will be sent along the objective
                with the same list index.
            prepended_conversations (Optional[List[List[Message]]]): Conversations to prepend to each
                objective. If provided, must match the length of objectives. Conversation will be sent along the
                objective with the same list index.
            memory_labels (Optional[Dict[str, str]]): Additional labels that can be applied to the prompts.
                The labels will be the same across all executions.
            return_partial_on_failure (bool): If True, returns AttackExecutorResult with completed results
                even when some objectives don't complete execution. If False, raises the first exception encountered.
                Defaults to False (raise on failure).
            **attack_params: Additional parameters specific to the attack strategy.

        Returns:
            AttackExecutorResult[AttackStrategyResultT]: Result container with completed results and
                any incomplete objectives. The result is iterable and behaves like a list of completed results.

        Raises:
            BaseException: If return_partial_on_failure=False and any objective doesn't complete execution.
            TypeError: If the attack strategy does not use MultiTurnAttackContext.

        Example:
            >>> executor = AttackExecutor(max_concurrency=3)
            >>> results = await executor.execute_multi_turn_attacks_async(
            ...     attack=multi_turn_attack,
            ...     objectives=["how to make a Molotov cocktail", "how to escalate privileges"],
            ...     messages=[Message(...), Message(...)]
            ... )
        """
        logger.warning(
            "execute_multi_turn_attacks_async is deprecated and will disappear in 0.13.0. "
            "Use execute_attack_async instead."
        )

        # Validate that the attack uses MultiTurnAttackContext
        if hasattr(attack, "_context_type") and not issubclass(attack._context_type, MultiTurnAttackContext):
            raise TypeError(
                f"Attack strategy {attack.__class__.__name__} must use MultiTurnAttackContext or a subclass of it."
            )

        return await self.execute_attack_async(
            attack=attack,
            objectives=objectives,
            prepended_conversations=cast(Optional[List[Optional[List[Message]]]], prepended_conversations),
            next_messages=cast(Optional[List[Optional[Message]]], messages),
            memory_labels=memory_labels,
            return_partial_on_failure=return_partial_on_failure,
            **attack_params,
        )

    def _validate_attack_batch_parameters(
        self,
        *,
        objectives: List[str],
        optional_list: Optional[List[Any]] = None,
        optional_list_name: str = "optional_list",
        prepended_conversations: Optional[List[List[Message]]] = None,
    ) -> None:
        """
        Validate common parameters for batch attack execution methods.

        Args:
            objectives (List[str]): List of attack objectives to test.
            optional_list (Optional[List[any]]): Optional list parameter to validate length against objectives.
            optional_list_name (str): Name of the optional list parameter for error messages.
            prepended_conversations (Optional[List[List[Message]]]): Conversations to prepend.

        Raises:
            ValueError: If validation fails.
        """
        # Validate input parameters
        if not objectives:
            raise ValueError("At least one objective must be provided")

        # Validate optional_list length if provided
        if optional_list is not None and len(optional_list) != len(objectives):
            raise ValueError(
                f"Number of {optional_list_name} ({len(optional_list)}) must"
                f" match number of objectives ({len(objectives)})"
            )

        # Validate prepended_conversations length if provided
        if prepended_conversations is not None and len(prepended_conversations) != len(objectives):
            raise ValueError(
                f"Number of prepended_conversations ({len(prepended_conversations)}) must match "
                f"number of objectives ({len(objectives)})"
            )

    async def execute_multi_objective_attack_with_context_async(
        self,
        *,
        attack: AttackStrategy[AttackStrategyContextT, AttackStrategyResultT],
        context_template: AttackStrategyContextT,
        objectives: List[str],
        return_partial_on_failure: bool = False,
    ) -> AttackExecutorResult[AttackStrategyResultT]:
        """
        Execute the same attack strategy with multiple objectives using context objects.

        .. deprecated::
            Use :meth:`execute_attack_async` instead. This method will be removed in a future version.

        This method works with context objects directly, duplicating the template context
        for each objective. Use this when you need fine-grained control over the context
        or have an existing context template to reuse.

        Args:
            attack (AttackStrategy[AttackStrategyContextT, AttackStrategyResultT]): The attack strategy to use for
                all objectives.
            context_template (AttackStrategyContextT): Template context that will be duplicated for each objective.
                Must have a 'duplicate()' method and an 'objective' attribute.
            objectives (List[str]): List of attack objectives to test. Each objective will be
                executed as a separate attack using a copy of the context template.
            return_partial_on_failure (bool): If True, returns AttackExecutorResult with completed results
                even when some objectives don't complete execution. If False, raises the first exception encountered.
                Defaults to False (raise on failure).

        Returns:
            AttackExecutorResult[AttackStrategyResultT]: Result container with completed results and
                any incomplete objectives. The result is iterable and behaves like a list of completed results.
                Use .has_incomplete or .raise_if_incomplete() to handle failures.

        Raises:
            AttributeError: If the context_template doesn't have required 'duplicate()' method
                or 'objective' attribute.
            BaseException: If return_partial_on_failure=False and any objective doesn't complete execution.

        Example:
            >>> executor = AttackExecutor(max_concurrency=3)
            >>> context = MultiTurnAttackContext(max_turns=5, ...)
            >>> results = await executor.execute_multi_objective_attack_with_context_async(
            ...     attack=prompt_injection_attack,
            ...     context_template=context,
            ...     objectives=["how to make a Molotov cocktail", "how to escalate privileges"]
            ... )
            >>> # Iterate directly over results
            >>> for result in results:
            ...     print(result)
        """
        logger.warning(
            "execute_multi_objective_attack_with_context_async is deprecated and will disappear in 0.13.0. "
            "Use execute_attack_async instead."
        )

        semaphore = asyncio.Semaphore(self._max_concurrency)

        async def execute_with_semaphore(ctx: AttackStrategyContextT) -> AttackStrategyResultT:
            async with semaphore:
                return await attack.execute_with_context_async(context=ctx)

        contexts = []
        for objective in objectives:
            # Create a deep copy of the context using its duplicate method
            context = context_template.duplicate()
            # Set the new objective (all attack contexts have objectives)
            context.objective = objective
            contexts.append(context)

        # Execute all tasks in parallel with concurrency control
        tasks = [execute_with_semaphore(ctx) for ctx in contexts]
        results_or_exceptions = await asyncio.gather(*tasks, return_exceptions=True)

        return self._process_execution_results(
            objectives=objectives,
            results_or_exceptions=results_or_exceptions,
            return_partial_on_failure=return_partial_on_failure,
        )

    def _process_execution_results(
        self,
        *,
        objectives: Sequence[str],
        results_or_exceptions: List[Union[AttackStrategyResultT, BaseException]],
        return_partial_on_failure: bool,
    ) -> AttackExecutorResult[AttackStrategyResultT]:
        """
        Process results from parallel execution, separating completed from incomplete objectives.

        Args:
            objectives (List[str]): List of objectives that were executed.
            results_or_exceptions (List[Union[AttackStrategyResultT, BaseException]]): Results from asyncio.gather
                with return_exceptions=True.
            return_partial_on_failure (bool): If True, returns AttackExecutorResult even when some objectives
                don't complete. If False, raises the first exception encountered.

        Returns:
            AttackExecutorResult[AttackStrategyResultT]: Result container with completed results and
                any incomplete objectives. Always returns this type regardless of success/failure status.

        Raises:
            BaseException: If return_partial_on_failure=False and any objective doesn't complete execution.
        """
        # Separate completed results from exceptions
        completed_results: List[AttackStrategyResultT] = []
        incomplete_objectives: List[tuple[str, BaseException]] = []

        for objective, result_or_exception in zip(objectives, results_or_exceptions):
            if isinstance(result_or_exception, BaseException):
                incomplete_objectives.append((objective, result_or_exception))
            else:
                completed_results.append(result_or_exception)  # type: ignore[arg-type]

        # If some incomplete and return_partial_on_failure is False, raise the first exception
        if incomplete_objectives and not return_partial_on_failure:
            raise incomplete_objectives[0][1]

        return AttackExecutorResult(completed_results=completed_results, incomplete_objectives=incomplete_objectives)
