# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging
from typing import Optional

from pyrit.attacks.base.attack_strategy import AttackStrategy
from pyrit.attacks.base.context import SingleTurnAttackContext
from pyrit.attacks.base.result import AttackResult
from pyrit.attacks.components.conversation_manager import ConversationManager
from pyrit.common.utils import combine_dict
from pyrit.models import (
    PromptRequestResponse,
    Score,
    SeedPrompt,
    SeedPromptGroup,
)
from pyrit.models.literals import ChatMessageRole
from pyrit.prompt_normalizer import PromptConverterConfiguration, PromptNormalizer
from pyrit.prompt_target import PromptTarget
from pyrit.score import Scorer

logger = logging.getLogger(__name__)


class PromptInjectionAttack(AttackStrategy[SingleTurnAttackContext, AttackResult]):
    """
    Implementation of single-turn prompt injection attack strategy.

    This class orchestrates a single-turn attack where malicious prompts are injected
    to try to achieve a specific objective against a target system. The strategy evaluates
    the target response using optional scorers to determine if the objective has been met.

    The attack flow consists of:
    1. Preparing the injection prompt based on the objective
    2. Sending the prompt to the target system through optional converters
    3. Evaluating the response with scorers if configured
    4. Retrying on failure up to the configured number of retries
    5. Returning the attack result with achievement status

    The strategy supports customization through prepended conversations, converters,
    and multiple scorer types for comprehensive evaluation.
    """

    def __init__(
        self,
        *,
        objective_target: PromptTarget,
        request_converter_configurations: Optional[list[PromptConverterConfiguration]] = None,
        response_converter_configurations: Optional[list[PromptConverterConfiguration]] = None,
        objective_scorer: Optional[Scorer] = None,
        auxiliary_scorers: Optional[list[Scorer]] = None,
        prompt_normalizer: Optional[PromptNormalizer] = None,
    ) -> None:
        """
        Initialize the prompt injection attack strategy.

        Args:
            objective_target (PromptTarget): The target to send prompts to
            request_converter_configurations (Optional[list[PromptConverterConfiguration]]):
                                                        Configurations for request converters
            response_converter_configurations (Optional[list[PromptConverterConfiguration]]):
                                                        Configurations for response converters
            objective_scorer (Optional[Scorer]): Scorer to evaluate if the objective was achieved
            auxiliary_scorers (Optional[list[Scorer]]): Additional scorers to evaluate the response
            prompt_normalizer (Optional[PromptNormalizer]): The prompt normalizer to use for sending prompts
        """
        super().__init__(logger=logger)

        # Skip criteria could be set directly in the injected prompt normalizer
        self._prompt_normalizer = prompt_normalizer or PromptNormalizer()

        if objective_scorer and objective_scorer.scorer_type != "true_false":
            raise ValueError("Objective scorer must be a true/false scorer")

        self._objective_target = objective_target
        self._objective_scorer = objective_scorer
        self._auxiliary_scorers = auxiliary_scorers or []

        self._request_converter_configurations = request_converter_configurations or []
        self._response_converter_configurations = response_converter_configurations or []

        self._conversation_manager = ConversationManager(
            attack_identifier=self.get_identifier(),
            prompt_normalizer=self._prompt_normalizer,
        )

    def _validate_context(self, *, context: SingleTurnAttackContext) -> None:
        """
        Validate the context before executing the attack.

        Args:
            context (SingleTurnAttackContext): The attack context containing parameters and objective

        Raises:
            ValueError: If the context is invalid
        """
        if not context.objective:
            raise ValueError("Attack objective must be provided in the context")

        if not context.conversation_id:
            raise ValueError("Conversation ID must be provided in the context")

        if context.num_retries_on_failure < 0:
            raise ValueError("Number of retries on failure must be non-negative")

    async def _setup_async(self, *, context: SingleTurnAttackContext) -> None:
        """
        Set up the attack by preparing conversation context.

        Args:
            context (SingleTurnAttackContext): The attack context containing attack parameters
        """
        # Initialize achieved_objective to False
        context.achieved_objective = False

        # Combine memory labels from context and attack strategy
        context.memory_labels = combine_dict(self._memory_labels, context.memory_labels)

        # Process prepended conversation if provided
        await self._conversation_manager.update_conversation_state_async(
            conversation_id=context.conversation_id,
            prepended_conversation=context.prepended_conversation,
            converter_configurations=self._request_converter_configurations,
        )

    async def _perform_attack_async(self, *, context: SingleTurnAttackContext) -> AttackResult:
        """
        Perform the prompt injection attack.

        Args:
            context: The attack context with objective and parameters

        Returns:
            AttackResult containing the outcome of the attack
        """
        # Log the attack configuration
        logger.info(f"Starting prompt injection attack with objective: {context.objective}")
        logger.info(f"Retries on failure: {context.num_retries_on_failure}")

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
        for attempt in range(context.num_retries_on_failure + 1):
            self._logger.debug(f"Attempt {attempt+1}/{context.num_retries_on_failure+1}")

            # Send the prompt
            response = await self._send_prompt_to_target_async(prompt_group=prompt_group, context=context)
            if not response:
                self._logger.warning(f"No response received on attempt {attempt+1} (likely filtered)")
                continue  # Retry if no response (filtered or error)

            # If no objective scorer, we are done after getting a response
            if not self._objective_scorer:
                break

            # Score the response
            score = await self._evaluate_response_async(response=response, objective=context.objective)

            # On success, return immediately
            context.achieved_objective = bool(score and score.get_value())
            if context.achieved_objective:
                break

        # Log the result of the attack
        self._log_objective_status(context)

        # Build and return result
        return AttackResult(
            conversation_id=context.conversation_id,
            objective=context.objective,
            orchestrator_identifier=self.get_identifier(),
            last_response=response.get_piece() if response else None,
            last_score=score,
            achieved_objective=context.achieved_objective,
            executed_turns=1,
        )

    async def _teardown_async(self, *, context: SingleTurnAttackContext) -> None:
        """Clean up after attack execution"""
        # Nothing to be done here, no-op
        pass

    def _get_prompt_group(self, context: SingleTurnAttackContext) -> SeedPromptGroup:
        """
        Prepare the seed prompt group based on the context.

        Args:
            context (SingleTurnAttackContext): The attack context containing the objective

        Returns:
            SeedPromptGroup: The seed prompt group containing the seed prompt
        """
        if context.seed_prompt_group:
            return context.seed_prompt_group

        return SeedPromptGroup(prompts=[SeedPrompt(value=context.objective, data_type="text")])

    async def _send_prompt_to_target_async(
        self, *, prompt_group: SeedPromptGroup, context: SingleTurnAttackContext
    ) -> Optional[PromptRequestResponse]:
        """
        Send the prompt to the target and return the response.

        Args:
            prompt_group (SeedPromptGroup): The seed prompt group to send
            context (SingleTurnAttackContext): The attack context containing parameters and labels

        Returns:
            Optional[PromptRequestResponse]: The model's response, or None if filtered out
        """

        return await self._prompt_normalizer.send_prompt_async(
            seed_prompt_group=prompt_group,
            target=self._objective_target,
            conversation_id=context.conversation_id,
            request_converter_configurations=self._request_converter_configurations,
            response_converter_configurations=self._response_converter_configurations,
            labels=context.memory_labels,  # combined with strategy labels at _setup()
            orchestrator_identifier=self.get_identifier(),
        )

    async def _evaluate_response_async(self, *, response: PromptRequestResponse, objective: str) -> Optional[Score]:
        """
        Evaluate the response against the objective using the configured scorers.

        Args:
            response (PromptRequestResponse): The response from the model
            objective (str): The natural-language description of the attack's objective

        Returns:
            Optional[Score]: The score assigned to the response, or None if no scoring was performed
        """

        role: ChatMessageRole = "assistant"

        # Run auxiliary scorers (no return value needed)
        if self._auxiliary_scorers:
            await Scorer.score_response_async(response=response, scorers=self._auxiliary_scorers, role_filter=role)

        # Run objective scorer
        if self._objective_scorer:
            return await Scorer.score_response_until_success_async(
                response=response, scorers=[self._objective_scorer], role_filter=role, task=objective
            )

        return None

    def _log_objective_status(self, context: SingleTurnAttackContext) -> None:
        """
        Log the status of the objective after the attack execution.

        Args:
            context (SingleTurnAttackContext): The attack context containing the objective and result
        """
        if context.achieved_objective:
            logger.info("Prompt injection attack achieved the objective")
        else:
            logger.info(
                f"The prompt injection attack has not achieved the objective after "
                f"{context.num_retries_on_failure + 1} attempts."
            )
