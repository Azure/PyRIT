# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import asyncio
import logging
import uuid
from typing import Any, Optional, Sequence

from pyrit.common.utils import combine_dict
from pyrit.models import (
    PromptRequestResponse,
    SeedPrompt,
    SeedPromptGroup,
)
from pyrit.models.filter_criteria import PromptConverterState, PromptFilterCriteria
from pyrit.orchestrator import (
    Orchestrator,
    OrchestratorResult,
    OrchestratorResultStatus,
)
from pyrit.prompt_normalizer import PromptConverterConfiguration, PromptNormalizer
from pyrit.prompt_target import PromptChatTarget, PromptTarget
from pyrit.prompt_target.batch_helper import batch_task_async
from pyrit.score import Scorer

logger = logging.getLogger(__name__)


class PromptSendingOrchestrator(Orchestrator):
    """
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

    def set_skip_criteria(
        self, *, skip_criteria: PromptFilterCriteria, skip_value_type: PromptConverterState = "original"
    ):
        """
        Sets the skip criteria for the orchestrator.

        If prompts match this in memory, then they won't be sent to a target.
        """
        self._prompt_normalizer.set_skip_criteria(skip_criteria=skip_criteria, skip_value_type=skip_value_type)

    async def _add_prepended_conversation_to_memory(
        self,
        prepended_conversation: Optional[list[PromptRequestResponse]],
        conversation_id: str,
    ):
        """
        Processes the prepended conversation by converting it if needed and adding it to memory.

        Args:
            prepended_conversation (Optional[list[PromptRequestResponse]]): The conversation to prepend
            conversation_id (str): The conversation ID to use for the request pieces
        """
        if not prepended_conversation:
            return

        if not isinstance(self._objective_target, PromptChatTarget):
            raise ValueError("Prepended conversation can only be used with a PromptChatTarget")

        await self._prompt_normalizer.add_prepended_conversation_to_memory(
            prepended_conversation=prepended_conversation,
            conversation_id=conversation_id,
            should_convert=self._should_convert_prepended_conversation,
            converter_configurations=self._request_converter_configurations,
            orchestrator_identifier=self.get_identifier(),
        )

    async def _score_auxiliary_async(self, result: PromptRequestResponse) -> None:
        """
        Scores the response using auxiliary scorers if they are configured.

        Args:
            result (PromptRequestResponse): The response to score
        """
        if not self._auxiliary_scorers:
            return

        tasks = []
        for piece in result.request_pieces:
            if piece.role == "assistant":
                for scorer in self._auxiliary_scorers:
                    tasks.append(scorer.score_async(request_response=piece))

        if tasks:
            await asyncio.gather(*tasks)

    async def _score_objective_async(
        self, result: PromptRequestResponse, objective: str
    ) -> tuple[OrchestratorResultStatus, Optional[Any]]:
        """
        Scores the response using the objective scorer if configured.

        Args:
            result (PromptRequestResponse): The response to score
            objective (str): The objective to score against

        Returns:
            tuple[OrchestratorResultStatus, Optional[Any]]: A tuple containing the status and objective score
            If the objective_scorer returns a list of scores, the first score that is true will be returned as the
            objective score.
            Note, this behavior can be overridden by setting the objective_scorer to a CompositeScorer.
        """
        if not self._objective_scorer:
            return "unknown", None

        status: OrchestratorResultStatus = "failure"
        objective_score = None
        first_failure_score = None

        for piece in result.request_pieces:
            if piece.role == "assistant":
                objective_score_list = await self._objective_scorer.score_async(
                    request_response=piece,
                    task=objective,
                )

                # Find and save the first score that is true
                for score in objective_score_list:
                    if score.get_value():
                        objective_score = score
                        status = "success"
                        break
                    elif first_failure_score is None:
                        first_failure_score = score
                if status == "success":
                    break

        # If no success was found, use the first failure score
        if status == "failure" and first_failure_score is not None:
            objective_score = first_failure_score

        return status, objective_score

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

        conversation_id = ""

        if not seed_prompt:
            seed_prompt = SeedPromptGroup(prompts=[SeedPrompt(value=objective)])

        status: OrchestratorResultStatus = "unknown"
        objective_score = None

        for _ in range(self._retries_on_objective_failure + 1):
            conversation_id = str(uuid.uuid4())
            await self._add_prepended_conversation_to_memory(prepended_conversation, conversation_id)

            result = await self._prompt_normalizer.send_prompt_async(
                seed_prompt_group=seed_prompt,
                target=self._objective_target,
                conversation_id=conversation_id,
                request_converter_configurations=self._request_converter_configurations,
                response_converter_configurations=self._response_converter_configurations,
                labels=combine_dict(existing_dict=self._global_memory_labels, new_dict=memory_labels),
                orchestrator_identifier=self.get_identifier(),
            )

            if not result:
                # This can happen if we skipped the prompts
                return None

            await self._score_auxiliary_async(result)

            status, objective_score = await self._score_objective_async(result, objective)

            if status == "success":
                break

        return OrchestratorResult(
            conversation_id=conversation_id,
            objective=objective,
            status=status,
            objective_score=objective_score,
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
