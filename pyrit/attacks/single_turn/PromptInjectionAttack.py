# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import asyncio
import logging
from typing import Optional, Tuple

from pyrit.attacks.base.attack_strategy import AttackStrategy
from pyrit.attacks.base.backtracking_strategy import BacktrackingStrategy
from pyrit.attacks.base.context import SingleTurnAttackContext
from pyrit.attacks.base.result import AttackResult
from pyrit.common.utils import combine_dict
from pyrit.models import (
    PromptRequestPiece,
    PromptRequestResponse,
    Score,
    SeedPrompt,
    SeedPromptGroup,
)
from pyrit.orchestrator.models.orchestrator_result import OrchestratorResultStatus
from pyrit.prompt_normalizer import PromptConverterConfiguration, PromptNormalizer
from pyrit.prompt_target import PromptChatTarget, PromptTarget
from pyrit.score import Scorer

logger = logging.getLogger(__name__)


class PromptInjectionAttack(AttackStrategy[SingleTurnAttackContext, AttackResult]):
    """
    Attack strategy that implements single-turn prompt injection attacks.

    This is a refactored implementation of the PromptSendingOrchestrator that follows
    the AttackStrategy pattern for better modularity and reuse.
    """

    def __init__(
        self,
        *,
        objective_target: PromptTarget,
        request_converter_configurations: Optional[list[PromptConverterConfiguration]] = None,
        response_converter_configurations: Optional[list[PromptConverterConfiguration]] = None,
        objective_scorer: Optional[Scorer] = None,
        auxiliary_scorers: Optional[list[Scorer]] = None,
        should_convert_prepended_conversation: bool = True,
        prompt_normalizer: Optional[PromptNormalizer] = None,
        backtracking_strategy: Optional[BacktrackingStrategy[SingleTurnAttackContext]] = None,
    ) -> None:
        """
        Initialize the prompt injection attack strategy.

        Args:
            objective_target: The target to send prompts to
            request_converter_configurations: Configurations for request converters
            response_converter_configurations: Configurations for response converters
            objective_scorer: Scorer to evaluate if the objective was achieved
            auxiliary_scorers: Additional scorers to evaluate the response
            should_convert_prepended_conversation: Whether to convert prepended conversations
            skip_criteria: Criteria to skip prompts
            skip_value_type: Type of value to check against skip criteria
            backtracking_strategy: Strategy for backtracking if needed
        """
        super().__init__(backtracking_strategy=backtracking_strategy)

        # Skip criteria could be set directly in the injected prompt normalizer
        self._prompt_normalizer = prompt_normalizer or PromptNormalizer()

        if objective_scorer and objective_scorer.scorer_type != "true_false":
            raise ValueError("Objective scorer must be a true/false scorer")

        self._objective_target = objective_target
        self._objective_scorer = objective_scorer
        self._auxiliary_scorers = auxiliary_scorers or []

        self._request_converter_configurations = request_converter_configurations or []
        self._response_converter_configurations = response_converter_configurations or []

        self._should_convert_prepended_conversation = should_convert_prepended_conversation

    async def _setup(self, *, context: SingleTurnAttackContext) -> None:
        """
        Set up the attack by preparing conversation context.

        Args:
            context: The attack context containing attack parameters
        """
        # Process prepended conversation if provided
        if context.prepended_conversation:
            if not isinstance(self._objective_target, PromptChatTarget):
                raise ValueError("Prepended conversation can only be used with a PromptChatTarget")

            await self._prompt_normalizer.add_prepended_conversation_to_memory(
                prepended_conversation=context.prepended_conversation,
                conversation_id=context.conversation_id,
                should_convert=self._should_convert_prepended_conversation,
                converter_configurations=self._request_converter_configurations,
                orchestrator_identifier=self.get_identifier(),
            )

    async def _perform_attack(self, *, context: SingleTurnAttackContext) -> AttackResult:
        """
        Perform the prompt injection attack.

        Args:
            context: The attack context with objective and parameters

        Returns:
            AttackResult containing the outcome of the attack
        """
        # Create seed prompt from context if not provided
        seed_prompt = context.seed_prompt_group
        if not seed_prompt:
            seed_prompt = SeedPromptGroup(
                prompts=[
                    SeedPrompt(
                        value=context.objective,
                        data_type="text",
                        metadata={"response_format": context.response_format} if context.response_format else {},
                    )
                ]
            )

        status: OrchestratorResultStatus = "unknown"
        objective_score = None
        last_response_piece = None

        # Retry logic - attempt multiple times if objective isn't met
        for _ in range(context.num_retries_on_failure + 1):
            result = await self._prompt_normalizer.send_prompt_async(
                seed_prompt_group=seed_prompt,
                target=self._objective_target,
                conversation_id=context.conversation_id,
                request_converter_configurations=self._request_converter_configurations,
                response_converter_configurations=self._response_converter_configurations,
                labels=combine_dict(existing_dict=self._memory_labels, new_dict=context.memory_labels),
                orchestrator_identifier=self.get_identifier(),
            )

            if not result:
                # This can happen if we skipped the prompts based on skip criteria
                continue

            # Score with auxiliary scorers
            await self._score_auxiliary_async(result)

            # Score against objective
            status, objective_score = await self._score_objective_async(result, context.objective)
            last_response_piece = result.get_piece()

            if status == "success":
                break

        # Create and return the attack result
        return self._create_result(
            context=context, status=status, objective_score=objective_score, last_response=last_response_piece
        )

    async def _teardown(self, *, context: SingleTurnAttackContext) -> None:
        """
        Clean up after attack execution.

        Args:
            context: The attack context
        """
        pass

    async def _score_auxiliary_async(self, result: PromptRequestResponse) -> None:
        """
        Score the response using auxiliary scorers.

        Args:
            result: The response to score
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
    ) -> Tuple[OrchestratorResultStatus, Optional[Score]]:
        """
        Score the response against the objective.

        Args:
            result: The response to score
            objective: The objective to score against

        Returns:
            A tuple of (status, score) where status is success, failure, or unknown
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

    def _create_result(
        self,
        *,
        context: SingleTurnAttackContext,
        status: OrchestratorResultStatus = "unknown",
        objective_score: Optional[Score] = None,
        last_response: Optional[PromptRequestPiece] = None,
    ) -> AttackResult:
        """
        Create an attack result from the context and outcome.

        Args:
            context: The attack context
            status: The status of the attack (success, failure, unknown)
            objective_score: Score assigned by the objective scorer
            last_response: The last response from the target

        Returns:
            An AttackResult representing the attack outcome
        """
        return AttackResult(
            conversation_id=context.conversation_id,
            objective=context.objective,
            orchestrator_identifier=self.get_identifier(),
            last_response=last_response,
            last_score=objective_score,
            achieved_objective=(status == "success"),
            executed_turns=1,  # Single-turn attack always executes 1 turn
        )
