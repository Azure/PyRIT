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
    Attack strategy that implements single-turn prompt injection attacks.
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
            objective_target: The target to send prompts to
            request_converter_configurations: Configurations for request converters
            response_converter_configurations: Configurations for response converters
            objective_scorer: Scorer to evaluate if the objective was achieved
            auxiliary_scorers: Additional scorers to evaluate the response
            should_convert_prepended_conversation: Whether to convert prepended conversations
            skip_criteria: Criteria to skip prompts
            skip_value_type: Type of value to check against skip criteria
        """
        super().__init__()

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
            context: The attack context containing parameters and objective

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
            context: The attack context containing attack parameters
        """

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
        response = None
        score = None

        for attempt in range(context.num_retries_on_failure + 1):
            self._logger.debug(f"Attempt {attempt+1}/{context.num_retries_on_failure+1}")
            try:
                # Send the prompt
                response = await self._send_prompt_async(prompt_group=prompt_group, context=context)

                if not response:
                    self._logger.warning(f"No response received on attempt {attempt+1} (likely filtered)")
                    continue  # Retry if no response (filtered or error)

                # If no objective scorer, we are done after getting a response
                if not self._objective_scorer:
                    break

                # Score the response
                score = await self._evaluate_response_async(response=response, objective=context.objective)

                # on success, return immediately
                success = score is not None and score.get_value()
                if success:
                    self._logger.info(f"Objective achieved on attempt {attempt+1}")
                    break

            except Exception as e:
                self._logger.error(f"Error on attempt {attempt+1}: {str(e)}", exc_info=True)

                # On last attempt, decide whether to fail or use partial results
                if attempt == context.num_retries_on_failure:
                    if response:
                        self._logger.warning("Using partial results from last successful response")
                        break
                    else:
                        raise

                # Otherwise, continue to next attempt
                continue

        # Build and return result
        return AttackResult(
            conversation_id=context.conversation_id,
            objective=context.objective,
            orchestrator_identifier=self.get_identifier(),
            last_response=response.get_piece() if response else None,
            last_score=score,
            achieved_objective=bool(score and score.get_value()),
            executed_turns=1,
        )

    async def _teardown_async(self, *, context: SingleTurnAttackContext) -> None:
        pass

    def _get_prompt_group(self, context: SingleTurnAttackContext) -> SeedPromptGroup:
        """
        Prepare the seed prompt group based on the context.

        Args:
            context: The attack context containing the objective

        Returns:
            SeedPromptGroup containing the seed prompt
        """
        if context.seed_prompt_group:
            return context.seed_prompt_group

        return SeedPromptGroup(prompts=[SeedPrompt(value=context.objective, data_type="text")])

    async def _send_prompt_async(
        self, *, prompt_group: SeedPromptGroup, context: SingleTurnAttackContext
    ) -> Optional[PromptRequestResponse]:
        """
        Send the prompt to the target and return the response.

        Args:
            prompt_group: The seed prompt group to send
            context: The attack context containing parameters and labels

        Returns:
            PromptRequestResponse containing the model's response, or None if filtered out
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
            response: The response from the model
            objective: The natural-language description of the attack's objective

        Returns:
            Score: The score assigned to the response, or None if no scoring was performed
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
