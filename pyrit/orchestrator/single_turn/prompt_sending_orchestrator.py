# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging
from typing import Any, LiteralString, Optional, Sequence, cast

from typing_extensions import deprecated

from pyrit.attacks import (
    AttackConverterConfig,
    AttackOutcome,
    AttackScoringConfig,
    PromptSendingAttack,
    SingleTurnAttackContext,
)
from pyrit.common import deprecation_message
from pyrit.models import (
    PromptRequestResponse,
    SeedPromptGroup,
)
from pyrit.models.filter_criteria import PromptConverterState, PromptFilterCriteria
from pyrit.orchestrator import (
    Orchestrator,
    OrchestratorResult,
    OrchestratorResultStatus,
)
from pyrit.prompt_normalizer import PromptConverterConfiguration, PromptNormalizer
from pyrit.prompt_target import PromptTarget
from pyrit.prompt_target.batch_helper import batch_task_async
from pyrit.score import Scorer

logger = logging.getLogger(__name__)


@deprecated(
    cast(
        LiteralString,
        deprecation_message(
            old_item="PromptSendingOrchestrator",
            new_item=PromptSendingAttack,
            removed_in="v0.12.0",
        ),
    ),
)
class PromptSendingOrchestrator(Orchestrator):
    """
    .. warning::
        `PromptSendingOrchestrator` is deprecated and will be removed in **v0.12.0**;
        use `pyrit.attacks.PromptSendingAttack` instead.

    This orchestrator takes a set of prompts, converts them using the list of PromptConverters,
    sends them to a target, and scores the resonses with scorers (if provided).
    """

    def __init__(
        self,
        objective_target: PromptTarget,
        request_converter_configurations: Optional[list[PromptConverterConfiguration]] = None,
        response_converter_configurations: Optional[list[PromptConverterConfiguration]] = None,
        objective_scorer: Optional[Scorer] = None,
        auxiliary_scorers: Optional[list[Scorer]] = None,
        should_convert_prepended_conversation: bool = True,
        batch_size: int = 10,
        retries_on_objective_failure: int = 0,
        verbose: bool = False,
    ) -> None:
        """
        Args:
            objective_target (PromptTarget): The target for sending prompts.
            prompt_converters (list[PromptConverter], Optional): List of prompt converters. These are stacked in
                the order they are provided. E.g. the output of converter1 is the input of converter2.
            scorers (list[Scorer], Optional): List of scorers to use for each prompt request response, to be
                scored immediately after receiving response. Default is None.
            batch_size (int, Optional): The (max) batch size for sending prompts. Defaults to 10.
                Note: If providing max requests per minute on the prompt_target, this should be set to 1 to
                ensure proper rate limit management.
            retries_on_objective_failure (int, Optional): Number of retries to attempt if objective fails. Defaults to
                0.
            verbose (bool, Optional): Whether to log debug information. Defaults to False.
        """
        super().__init__(verbose=verbose)

        self._prompt_normalizer = PromptNormalizer()

        if objective_scorer and objective_scorer.scorer_type != "true_false":
            raise ValueError("Objective scorer must be a true/false scorer")

        self._objective_scorer = objective_scorer or None
        self._auxiliary_scorers = auxiliary_scorers or []

        self._objective_target = objective_target

        self._request_converter_configurations = request_converter_configurations or []
        self._response_converter_configurations = response_converter_configurations or []

        self._should_convert_prepended_conversation = should_convert_prepended_conversation
        self._batch_size = batch_size
        self._retries_on_objective_failure = retries_on_objective_failure

        # Build the new attack model
        self._attack = PromptSendingAttack(
            objective_target=objective_target,
            attack_converter_config=AttackConverterConfig(
                request_converters=self._request_converter_configurations,
                response_converters=self._response_converter_configurations,
            ),
            attack_scoring_config=AttackScoringConfig(
                objective_scorer=objective_scorer,
                auxiliary_scorers=self._auxiliary_scorers,
            ),
            prompt_normalizer=self._prompt_normalizer,
        )

    def set_skip_criteria(
        self, *, skip_criteria: PromptFilterCriteria, skip_value_type: PromptConverterState = "original"
    ):
        """
        Sets the skip criteria for the orchestrator.

        If prompts match this in memory, then they won't be sent to a target.
        """
        self._prompt_normalizer.set_skip_criteria(skip_criteria=skip_criteria, skip_value_type=skip_value_type)

    async def run_attack_async(
        self,
        *,
        objective: str,
        seed_prompt: Optional[SeedPromptGroup] = None,
        prepended_conversation: Optional[list[PromptRequestResponse]] = None,
        memory_labels: Optional[dict[str, str]] = None,
    ) -> OrchestratorResult:
        """
        Runs the attack.

        Args:
            objective (str): The objective of the attack.
            seed_prompt (SeedPromptGroup, Optional): The seed prompt group to start the conversation. By default the
                objective is used.
            prepended_conversation (list[PromptRequestResponse], Optional): The conversation to prepend to the attack.
                Sent to objective target.
            memory_labels (dict[str, str], Optional): The memory labels to use for the attack.
        """

        context = SingleTurnAttackContext(
            objective=objective,
            seed_prompt_group=seed_prompt,
            prepended_conversation=prepended_conversation or [],
            max_attempts_on_failure=self._retries_on_objective_failure,
            memory_labels=memory_labels or {},
        )

        result = await self._attack.execute_async(context=context)

        # Map attack outcome to orchestrator status
        status_mapping: dict[AttackOutcome, OrchestratorResultStatus] = {
            AttackOutcome.SUCCESS: "success",
            AttackOutcome.FAILURE: "failure",
            AttackOutcome.UNDETERMINED: "unknown",
        }

        return OrchestratorResult(
            conversation_id=result.conversation_id,
            objective=objective,
            status=status_mapping.get(result.outcome, "unknown"),
            objective_score=result.last_score,
        )

    async def run_attacks_async(
        self,
        *,
        objectives: list[str],
        seed_prompts: Optional[list[SeedPromptGroup]] = None,
        prepended_conversations: Optional[list[list[PromptRequestResponse]]] = None,
        memory_labels: Optional[dict[str, str]] = None,
    ) -> list[OrchestratorResult]:
        """
        Runs multiple attacks in parallel using batch_size.

        Args:
            objectives (list[str]): List of objectives for the attacks.
            seed_prompts (list[SeedPromptGroup], Optional): List of seed prompt groups to start the conversations.
                If not provided, each objective will be used as its own seed prompt.
            prepended_conversation (list[PromptRequestResponse], Optional): The conversation to prepend to each attack.
            memory_labels (dict[str, str], Optional): The memory labels to use for the attacks.
        Returns:
            list[OrchestratorResult]: List of results from each attack.
        """
        if not seed_prompts:
            seed_prompts = [None] * len(objectives)
        elif len(seed_prompts) != len(objectives):
            raise ValueError("Number of seed prompts must match number of objectives")

        if not prepended_conversations:
            prepended_conversations = [None] * len(objectives)
        elif len(prepended_conversations) != len(objectives):
            raise ValueError("Number of prepended conversations must match number of objectives")

        batch_items: list[Sequence[Any]] = [objectives, seed_prompts, prepended_conversations]

        batch_item_keys = [
            "objective",
            "seed_prompt",
            "prepended_conversation",
        ]

        results = await batch_task_async(
            prompt_target=self._objective_target,
            batch_size=self._batch_size,
            items_to_batch=batch_items,
            task_func=self.run_attack_async,
            task_arguments=batch_item_keys,
            memory_labels=memory_labels,
        )

        return [result for result in results if result is not None]

    async def _run_attacks_with_only_objectives_async(
        self,
        *,
        objectives: list[str],
        memory_labels: Optional[dict[str, str]] = None,
    ) -> list[OrchestratorResult]:
        """
        Runs multiple role play attacks in parallel using batch_size.

        Args:
            objectives (list[str]): List of objectives for the attacks.
            memory_labels (dict[str, str], Optional): The memory labels to use for the attacks.
        Returns:
            list[OrchestratorResult]: List of results from each attack.
        """

        batch_items = [
            objectives,
        ]

        batch_item_keys = [
            "objective",
        ]

        results = await batch_task_async(
            prompt_target=self._objective_target,
            batch_size=self._batch_size,
            items_to_batch=batch_items,
            task_func=self.run_attack_async,
            task_arguments=batch_item_keys,
            memory_labels=memory_labels,
        )

        return results
