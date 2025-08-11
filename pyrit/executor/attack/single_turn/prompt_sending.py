# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging
import uuid
from typing import Optional

from pyrit.common.utils import combine_dict, warn_if_set
from pyrit.executor.attack.component import ConversationManager
from pyrit.executor.attack.core import AttackConverterConfig, AttackScoringConfig
from pyrit.executor.attack.single_turn.single_turn_attack_strategy import (
    SingleTurnAttackContext,
    SingleTurnAttackStrategy,
)
from pyrit.models import (
    AttackOutcome,
    AttackResult,
    ConversationReference,
    ConversationType,
    PromptRequestResponse,
    Score,
    SeedPrompt,
    SeedPromptGroup,
)
from pyrit.prompt_normalizer import PromptNormalizer
from pyrit.prompt_target import PromptTarget
from pyrit.score import Scorer

logger = logging.getLogger(__name__)


class PromptSendingAttack(SingleTurnAttackStrategy):
    """
    Implementation of single-turn prompt sending attack strategy.

    This class orchestrates a single-turn attack where malicious prompts are injected
    to try to achieve a specific objective against a target system. The strategy evaluates
    the target response using optional scorers to determine if the objective has been met.

    The attack flow consists of:
    1. Preparing the prompt based on the objective.
    2. Sending the prompt to the target system through optional converters.
    3. Evaluating the response with scorers if configured.
    4. Retrying on failure up to the configured number of retries.
    5. Returning the attack result with achievement status.

    The strategy supports customization through prepended conversations, converters,
    and multiple scorer types for comprehensive evaluation.
    """

    def __init__(
        self,
        *,
        objective_target: PromptTarget,
        attack_converter_config: Optional[AttackConverterConfig] = None,
        attack_scoring_config: Optional[AttackScoringConfig] = None,
        prompt_normalizer: Optional[PromptNormalizer] = None,
        max_attempts_on_failure: int = 0,
    ) -> None:
        """
        Initialize the prompt injection attack strategy.

        Args:
            objective_target (PromptTarget): The target system to attack.
            attack_converter_config (Optional[AttackConverterConfig]): Configuration for prompt converters.
            attack_scoring_config (Optional[AttackScoringConfig]): Configuration for scoring components.
            prompt_normalizer (Optional[PromptNormalizer]): Normalizer for handling prompts.
            max_attempts_on_failure (int): Maximum number of attempts to retry on failure.

        Raises:
            ValueError: If the objective scorer is not a true/false scorer.
        """
        # Initialize base class
        super().__init__(logger=logger, context_type=SingleTurnAttackContext)

        # Store the objective target
        self._objective_target = objective_target

        # Initialize the converter configuration
        attack_converter_config = attack_converter_config or AttackConverterConfig()
        self._request_converters = attack_converter_config.request_converters
        self._response_converters = attack_converter_config.response_converters

        # Initialize scoring configuration
        attack_scoring_config = attack_scoring_config or AttackScoringConfig()

        # Check for unused optional parameters and warn if they are set
        warn_if_set(config=attack_scoring_config, unused_fields=["refusal_scorer"], log=logger)

        self._auxiliary_scorers = attack_scoring_config.auxiliary_scorers
        self._objective_scorer = attack_scoring_config.objective_scorer
        if self._objective_scorer and self._objective_scorer.scorer_type != "true_false":
            raise ValueError("Objective scorer must be a true/false scorer")

        # Skip criteria could be set directly in the injected prompt normalizer
        self._prompt_normalizer = prompt_normalizer or PromptNormalizer()
        self._conversation_manager = ConversationManager(
            attack_identifier=self.get_identifier(),
            prompt_normalizer=self._prompt_normalizer,
        )

        # Set the maximum attempts on failure
        if max_attempts_on_failure < 0:
            raise ValueError("max_attempts_on_failure must be a non-negative integer")

        self._max_attempts_on_failure = max_attempts_on_failure

    def _validate_context(self, *, context: SingleTurnAttackContext) -> None:
        """
        Validate the context before executing the attack.

        Args:
            context (SingleTurnAttackContext): The attack context containing parameters and objective.

        Raises:
            ValueError: If the context is invalid.
        """
        if not context.objective or context.objective.isspace():
            raise ValueError("Attack objective must be provided and non-empty in the context")

    async def _setup_async(self, *, context: SingleTurnAttackContext) -> None:
        """
        Set up the attack by preparing conversation context.

        Args:
            context (SingleTurnAttackContext): The attack context containing attack parameters.
        """
        # Ensure the context has a conversation ID
        context.conversation_id = str(uuid.uuid4())

        # Combine memory labels from context and attack strategy
        context.memory_labels = combine_dict(self._memory_labels, context.memory_labels)

        # Process prepended conversation if provided
        await self._conversation_manager.update_conversation_state_async(
            target=self._objective_target,
            conversation_id=context.conversation_id,
            prepended_conversation=context.prepended_conversation,
            request_converters=self._request_converters,
            response_converters=self._response_converters,
        )

    async def _perform_async(self, *, context: SingleTurnAttackContext) -> AttackResult:
        """
        Perform the prompt injection attack.

        Args:
            context: The attack context with objective and parameters.

        Returns:
            AttackResult containing the outcome of the attack.
        """
        # Log the attack configuration
        self._logger.info(f"Starting {self.__class__.__name__} with objective: {context.objective}")
        self._logger.info(f"Max attempts: {self._max_attempts_on_failure}")

        # Execute with retries
        response = None
        score = None

        # Attack execution steps:
        # 1) Construct the seed prompt(s) that will be injected into the conversation
        # 2) Send the prompt to the target model using the prompt normalizer helper
        # 3) If the call fails or the response is filtered, retry as configured
        # 4) When a response is obtained, optionally evaluate it with the objective scorer
        # 5) Stop early if the objective is achieved; otherwise continue retry loop
        # 6) After retries are exhausted, compile the final response and score
        # 7) Return an AttackResult object that captures the outcome of the attack

        # Prepare the prompt
        prompt_group = self._get_prompt_group(context)

        # Execute with retries
        for attempt in range(self._max_attempts_on_failure + 1):
            self._logger.debug(f"Attempt {attempt+1}/{self._max_attempts_on_failure + 1}")

            # Send the prompt
            response = await self._send_prompt_to_objective_target_async(prompt_group=prompt_group, context=context)
            if not response:
                self._logger.warning(f"No response received on attempt {attempt+1} (likely filtered)")
                continue  # Retry if no response (filtered or error)

            # Score the response including auxiliary and objective scoring
            score = await self._evaluate_response_async(response=response, objective=context.objective)

            # If there is no objective, we have a response but can't determine success
            if not self._objective_scorer:
                break

            # On success, return immediately
            if bool(score and score.get_value()):
                break

            # On failure, store and create new conversation if there are more attempts remaining
            if attempt < self._max_attempts_on_failure:
                context.related_conversations.add(
                    ConversationReference(
                        conversation_id=context.conversation_id,
                        conversation_type=ConversationType.PRUNED,
                    )
                )
                await self._setup_async(context=context)  # Reset conversation for next attempt

        # Determine the outcome
        outcome, outcome_reason = self._determine_attack_outcome(response=response, score=score, context=context)

        result = AttackResult(
            conversation_id=context.conversation_id,
            objective=context.objective,
            attack_identifier=self.get_identifier(),
            last_response=response.get_piece() if response else None,
            last_score=score,
            related_conversations=context.related_conversations,
            outcome=outcome,
            outcome_reason=outcome_reason,
            executed_turns=1,
        )

        return result

    def _determine_attack_outcome(
        self, *, response: Optional[PromptRequestResponse], score: Optional[Score], context: SingleTurnAttackContext
    ) -> tuple[AttackOutcome, Optional[str]]:
        """
        Determine the outcome of the attack based on the response and score.

        Args:
            response (Optional[PromptRequestResponse]): The last response from the target (if any).
            score (Optional[Score]): The objective score (if any).
            context (SingleTurnAttackContext): The attack context containing configuration.

        Returns:
            tuple[AttackOutcome, Optional[str]]: A tuple of (outcome, outcome_reason).
        """
        if not self._objective_scorer:
            # No scorer means we can't determine success/failure
            return AttackOutcome.UNDETERMINED, "No objective scorer configured"

        if score and score.get_value():
            # We have a positive score, so it's a success
            return AttackOutcome.SUCCESS, "Objective achieved according to scorer"

        if response:
            # We got response(s) but none achieved the objective
            return (
                AttackOutcome.FAILURE,
                f"Failed to achieve objective after {self._max_attempts_on_failure + 1} attempts",
            )

        # No response at all (all attempts filtered/failed)
        return AttackOutcome.FAILURE, "All attempts were filtered or failed to get a response"

    async def _teardown_async(self, *, context: SingleTurnAttackContext) -> None:
        """Clean up after attack execution"""
        # Nothing to be done here, no-op
        pass

    def _get_prompt_group(self, context: SingleTurnAttackContext) -> SeedPromptGroup:
        """
        Prepare the seed prompt group for the attack.

        If a seed_prompt_group is provided in the context, it will be used directly.
        Otherwise, creates a new SeedPromptGroup with the objective as a text prompt.

        Args:
            context (SingleTurnAttackContext): The attack context containing the objective
                and optionally a pre-configured seed_prompt_group.

        Returns:
            SeedPromptGroup: The seed prompt group to be used in the attack.
        """
        if context.seed_prompt_group:
            return context.seed_prompt_group

        return SeedPromptGroup(prompts=[SeedPrompt(value=context.objective, data_type="text")])

    async def _send_prompt_to_objective_target_async(
        self, *, prompt_group: SeedPromptGroup, context: SingleTurnAttackContext
    ) -> Optional[PromptRequestResponse]:
        """
        Send the prompt to the target and return the response.

        Args:
            prompt_group (SeedPromptGroup): The seed prompt group to send.
            context (SingleTurnAttackContext): The attack context containing parameters and labels.

        Returns:
            Optional[PromptRequestResponse]: The model's response if successful, or None if
                the request was filtered, blocked, or encountered an error.
        """

        return await self._prompt_normalizer.send_prompt_async(
            seed_prompt_group=prompt_group,
            target=self._objective_target,
            conversation_id=context.conversation_id,
            request_converter_configurations=self._request_converters,
            response_converter_configurations=self._response_converters,
            labels=context.memory_labels,  # combined with strategy labels at _setup()
            orchestrator_identifier=self.get_identifier(),
        )

    async def _evaluate_response_async(self, *, response: PromptRequestResponse, objective: str) -> Optional[Score]:
        """
        Evaluate the response against the objective using the configured scorers.

        This method first runs all auxiliary scorers (if configured) to collect additional
        metrics, then runs the objective scorer to determine if the attack succeeded.

        Args:
            response (PromptRequestResponse): The response from the model.
            objective (str): The natural-language description of the attack's objective.

        Returns:
            Optional[Score]: The score from the objective scorer if configured, or None if
                no objective scorer is set. Note that auxiliary scorer results are not returned
                but are still executed and stored.
        """
        scoring_results = await Scorer.score_response_with_objective_async(
            response=response,
            auxiliary_scorers=self._auxiliary_scorers,
            objective_scorers=[self._objective_scorer] if self._objective_scorer else None,
            role_filter="assistant",
            task=objective,
        )

        objective_scores = scoring_results["objective_scores"]
        if not objective_scores:
            return None

        return objective_scores[0]
