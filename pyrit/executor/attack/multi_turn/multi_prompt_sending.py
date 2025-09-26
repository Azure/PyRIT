# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging
from dataclasses import dataclass, field
from typing import List, Optional

from pyrit.common.utils import combine_dict, get_kwarg_param
from pyrit.executor.attack.component import ConversationManager
from pyrit.executor.attack.core import (
    AttackConverterConfig,
    AttackScoringConfig,
)
from pyrit.executor.attack.multi_turn.multi_turn_attack_strategy import (
    ConversationSession,
    MultiTurnAttackContext,
    MultiTurnAttackStrategy,
)
from pyrit.models import (
    AttackOutcome,
    AttackResult,
    PromptRequestResponse,
    Score,
    SeedPrompt,
    SeedPromptGroup,
)
from pyrit.prompt_normalizer import PromptNormalizer
from pyrit.prompt_target import PromptTarget
from pyrit.score import Scorer

logger = logging.getLogger(__name__)


@dataclass
class MultiPromptSendingAttackContext(MultiTurnAttackContext):
    """Context for the MultiPromptSending attack strategy."""

    # Predefined prompt sequence to send to the target
    prompt_sequence: List[str] = field(default_factory=list)


class MultiPromptSendingAttack(MultiTurnAttackStrategy[MultiPromptSendingAttackContext, AttackResult]):
    """
    Implementation of multi-prompt sending attack strategy.

    This class orchestrates a multi-turn attack where a series of predefined malicious
    prompts are sent sequentially to try to achieve a specific objective against a target
    system. The strategy evaluates the final target response using optional scorers to
    determine if the objective has been met.

    The attack flow consists of:
    1. Sending each predefined prompt to the target system in sequence.
    2. Continuing until all predefined prompts are sent.
    3. Evaluating the final response with scorers if configured.
    4. Returning the attack result with achievement status.

    Note: This attack always runs all predefined prompts regardless of whether the
    objective is achieved early in the sequence.

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
    ) -> None:
        """
        Initialize the multi-prompt sending attack strategy.

        Args:
            objective_target (PromptTarget): The target system to attack.
            attack_converter_config (Optional[AttackConverterConfig]): Configuration for prompt converters.
            attack_scoring_config (Optional[AttackScoringConfig]): Configuration for scoring components.
            prompt_normalizer (Optional[PromptNormalizer]): Normalizer for handling prompts.

        Raises:
            ValueError: If the objective scorer is not a true/false scorer.
        """
        # Initialize base class
        super().__init__(logger=logger, context_type=MultiPromptSendingAttackContext)

        # Store the objective target
        self._objective_target = objective_target

        # Initialize the converter configuration
        attack_converter_config = attack_converter_config or AttackConverterConfig()
        self._request_converters = attack_converter_config.request_converters
        self._response_converters = attack_converter_config.response_converters

        # Initialize scoring configuration
        attack_scoring_config = attack_scoring_config or AttackScoringConfig()

        self._auxiliary_scorers = attack_scoring_config.auxiliary_scorers
        self._objective_scorer = attack_scoring_config.objective_scorer
        if self._objective_scorer and self._objective_scorer.scorer_type != "true_false":
            raise ValueError("Objective scorer must be a true/false scorer")

        # Initialize prompt normalizer and conversation manager
        self._prompt_normalizer = prompt_normalizer or PromptNormalizer()
        self._conversation_manager = ConversationManager(
            attack_identifier=self.get_identifier(),
            prompt_normalizer=self._prompt_normalizer,
        )

    def _validate_context(self, *, context: MultiPromptSendingAttackContext) -> None:
        """
        Validate the context before executing the attack.

        Args:
            context (MultiPromptSendingAttackContext): The attack context containing parameters and objective.

        Raises:
            ValueError: If the context is invalid.
        """
        if not context.objective or context.objective.isspace():
            raise ValueError("Attack objective must be provided and non-empty in the context")

        if not context.prompt_sequence or len(context.prompt_sequence) == 0:
            raise ValueError("Prompt sequence must be provided and non-empty in the context")

        if bool(list(filter(lambda x: not x or str.isspace(x), context.prompt_sequence))):
            raise ValueError("Prompt sequence must not contain empty prompts")

    async def _setup_async(self, *, context: MultiPromptSendingAttackContext) -> None:
        """
        Set up the attack by preparing conversation context.

        Args:
            context (MultiPromptSendingAttackContext): The attack context containing attack parameters.
        """
        # Ensure the context has a session (like red_teaming.py does)
        context.session = ConversationSession()

        # Combine memory labels from context and attack strategy
        context.memory_labels = combine_dict(self._memory_labels, context.memory_labels)

        # Process prepended conversation if provided
        await self._conversation_manager.update_conversation_state_async(
            target=self._objective_target,
            conversation_id=context.session.conversation_id,
            prepended_conversation=context.prepended_conversation,
            request_converters=self._request_converters,
            response_converters=self._response_converters,
        )

    async def _perform_async(self, *, context: MultiPromptSendingAttackContext) -> AttackResult:
        """
        Perform the multi-prompt sending attack.

        Args:
            context: The attack context with objective, predefined prompt sequence and parameters.

        Returns:
            AttackResult containing the outcome of the attack.
        """
        # Log the attack configuration
        logger.info(f"Starting {self.__class__.__name__} with objective: {context.objective}")

        # Attack execution steps:
        # 1) Send each predefined malicious prompt to the target sequentially
        # 2) Continue until all the predefined prompts are sent
        # 3) Score the final response using the configured objective scorer
        # 4) Return an AttackResult object that captures the outcome of the attack

        response = None
        score = None

        for prompt_index, prompt_text in enumerate(context.prompt_sequence):
            logger.info(f"Processing prompt {prompt_index + 1}/{len(context.prompt_sequence)}")
            logger.debug(f"Prompt content: {prompt_text}")

            # Create seed prompt group for this prompt
            prompt_group = SeedPromptGroup(prompts=[SeedPrompt(value=prompt_text, data_type="text")])

            # Send the prompt
            prompt_response = await self._send_prompt_to_objective_target_async(
                prompt_group=prompt_group, context=context
            )

            # Update context with latest response (may be None if sending failed)
            if prompt_response:
                response = prompt_response
                context.last_response = response
                context.executed_turns += 1
                self._logger.debug(f"Successfully sent prompt {prompt_index + 1}")
            else:
                response = None
                self._logger.warning(f"Failed to send prompt {prompt_index + 1}, terminating")
                break

        # Score the last response including auxiliary and objective scoring
        if response is not None:
            score = await self._evaluate_response_async(response=response, objective=context.objective)
        else:
            score = None

        # Determine the outcome
        outcome, outcome_reason = self._determine_attack_outcome(response=response, score=score, context=context)

        result = AttackResult(
            conversation_id=context.session.conversation_id,
            objective=context.objective,
            attack_identifier=self.get_identifier(),
            last_response=response.get_piece() if response else None,
            last_score=score,
            related_conversations=context.related_conversations,
            outcome=outcome,
            outcome_reason=outcome_reason,
            executed_turns=context.executed_turns,
        )

        return result

    def _determine_attack_outcome(
        self,
        *,
        response: Optional[PromptRequestResponse],
        score: Optional[Score],
        context: MultiPromptSendingAttackContext,
    ) -> tuple[AttackOutcome, Optional[str]]:
        """
        Determine the outcome of the attack based on the response and score.

        Args:
            response (Optional[PromptRequestResponse]): The last response from the target (if any).
            score (Optional[Score]): The objective score (if any).
            context (MultiPromptSendingAttackContext): The attack context containing configuration.

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
            # We got response(s) but the final response did not achieve the objective
            return (
                AttackOutcome.FAILURE,
                "Failed to achieve objective",
            )

        # At least one prompt was filtered or failed to get a response
        return AttackOutcome.FAILURE, "At least one prompt was filtered or failed to get a response"

    async def _teardown_async(self, *, context: MultiPromptSendingAttackContext) -> None:
        """Clean up after attack execution"""
        # Nothing to be done here, no-op
        pass

    async def _send_prompt_to_objective_target_async(
        self, *, prompt_group: SeedPromptGroup, context: MultiPromptSendingAttackContext
    ) -> Optional[PromptRequestResponse]:
        """
        Send the prompt to the target and return the response.

        Args:
            prompt_group (SeedPromptGroup): The seed prompt group to send.
            context (MultiPromptSendingAttackContext): The attack context containing parameters and labels.

        Returns:
            Optional[PromptRequestResponse]: The model's response if successful, or None if
                the request was filtered, blocked, or encountered an error.
        """
        return await self._prompt_normalizer.send_prompt_async(
            seed_prompt_group=prompt_group,
            target=self._objective_target,
            conversation_id=context.session.conversation_id,
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

    async def execute_async(
        self,
        **kwargs,
    ) -> AttackResult:
        """
        Execute the attack strategy asynchronously with the provided parameters.
        """

        # Validate parameters before creating context
        prompt_sequence = get_kwarg_param(
            kwargs=kwargs, param_name="prompt_sequence", expected_type=list, required=True
        )

        return await super().execute_async(**kwargs, prompt_sequence=prompt_sequence)
