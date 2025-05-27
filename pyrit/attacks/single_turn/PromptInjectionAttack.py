# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import asyncio
import logging
from typing import List, Optional, Tuple

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

    def _validate_context(self, *, context: SingleTurnAttackContext) -> None:
        """
        Validate the context before executing the attack.

        Args:
            context: The attack context containing parameters and objective

        Raises:
            ValueError: If the context is invalid
        """
        if not isinstance(self._objective_target, PromptChatTarget):
            raise ValueError("Objective target must be a PromptChatTarget for prompt injection attacks")

        if not context.objective:
            raise ValueError("Attack objective must be provided in the context")

        if not context.conversation_id:
            raise ValueError("Conversation ID must be provided in the context")

    async def _setup_async(self, *, context: SingleTurnAttackContext) -> None:
        """
        Set up the attack by preparing conversation context.

        Args:
            context: The attack context containing attack parameters
        """

        # Combine memory labels from context and attack strategy
        context.memory_labels = combine_dict(self._memory_labels, context.memory_labels)

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

    async def _perform_attack_async(self, *, context: SingleTurnAttackContext) -> AttackResult:
        """
        Perform the prompt injection attack.

        Args:
            context: The attack context with objective and parameters

        Returns:
            AttackResult containing the outcome of the attack
        """
       # Prepare the prompt
        seed_prompt = self._prepare_seed_prompt(context)
        
        # Execute with retries
        response, score = await self._execute_with_retries_async(
            seed_prompt=seed_prompt,
            context=context
        )
        
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
        """
        Clean up after attack execution.

        Args:
            context: The attack context
        """
        pass

    def _prepare_seed_prompt(self, context: SingleTurnAttackContext) -> SeedPromptGroup:
        """
        Prepare the seed prompt group based on the context.
        
        Args:
            context: The attack context containing the objective
        
        Returns:
            SeedPromptGroup containing the seed prompt
        """
        if context.seed_prompt_group:
            return context.seed_prompt_group
            
        return SeedPromptGroup(
            prompts=[
                SeedPrompt(
                    value=context.objective,
                    data_type="text"
                )
            ]
        )
    
    async def _execute_with_retries_async(
        self,
        *,
        seed_prompt: SeedPromptGroup,
        context: SingleTurnAttackContext
    ) -> Tuple[Optional[PromptRequestResponse], Optional[Score]]:
        """
        Execute the prompt injection attack with retries
        Args:
            seed_prompt: The seed prompt group to use for the attack
            context: The attack context containing parameters and objective
        Returns:
            Tuple containing the response and score, or None if no valid response was obtained
        """
        
        for attempt in range(context.num_retries_on_failure + 1):
            self._logger.debug(f"Attempt {attempt+1}/{context.num_retries_on_failure+1}")

            # Send the prompt
            response = await self._send_prompt_async(seed_prompt=seed_prompt, context=context)
            
            if not response:
                continue  # Skip if filtered
                
            # Score the response
            score = await self._evaluate_response_async(response=response, objective=context.objective)
            
            # Success - return immediately
            success = score is not None and score.get_value()
            if success:
                return response, score
                
            # Last attempt - return what we have
            if attempt == context.num_retries_on_failure:
                return response, score
        
        return None, None
    
    async def _send_prompt_async(
        self,
        *,
        seed_prompt: SeedPromptGroup,
        context: SingleTurnAttackContext
    ) -> Optional[PromptRequestResponse]:
        """
        Send the prompt to the target and return the response.

        Args:
            seed_prompt: The seed prompt group to send
            context: The attack context containing parameters and labels

        Returns:
            PromptRequestResponse containing the model's response, or None if filtered out
        """
        
        return await self._prompt_normalizer.send_prompt_async(
            seed_prompt_group=seed_prompt,
            target=self._objective_target,
            conversation_id=context.conversation_id,
            request_converter_configurations=self._request_converter_configurations,
            response_converter_configurations=self._response_converter_configurations,
            labels=context.memory_labels, # combined with strategy labels at _setup()
            orchestrator_identifier=self.get_identifier(),
        )
    
    async def _evaluate_response_async(
        self, 
        *,
        response: PromptRequestResponse, 
        objective: str
    ) -> Optional[Score]:
        """
        Evaluate the response against the objective using the configured scorers.
        
        Args:
            response: The response from the model
            objective: The natural-language description of the attack's objective
        
        Returns:
            Score: The score assigned to the response, or None if no scoring was performed
        """

        # Only score assistant responses
        assistant_responses = list(response.filter_by_role(role="assistant"))
        
        # Run auxiliary scorers (no return value needed)
        await self._run_auxiliary_scorers_async(assistant_responses=assistant_responses)
        
        # Run objective scorer
        return await self._run_objective_scorer_async(assistant_responses=assistant_responses, objective=objective)
    
    async def _run_auxiliary_scorers_async(self, *, assistant_responses: List[PromptRequestPiece]) -> None:
        """
        Run all auxiliary scorers on the assistant responses.

        Args:
            assistant_responses: List of PromptRequestPiece objects representing the assistant's responses
        
        Returns:
            None: This method does not return a value, it runs scorers asynchronously
        """
        if not self._auxiliary_scorers:
            return
        
        tasks = [
            scorer.score_async(request_response=piece)
            for piece in assistant_responses
            for scorer in self._auxiliary_scorers
        ]
        
        if tasks:
            await asyncio.gather(*tasks)

    async def _run_objective_scorer_async(
        self, 
        *,
        assistant_responses: List[PromptRequestPiece], 
        objective: str
    ) -> Optional[Score]:
        """
        Run the objective scorer on the assistant responses.

        Args:
            assistant_responses: List of PromptRequestPiece objects representing the assistant's responses
            objective: The natural-language description of the attack's objective

        Returns:
            Score: The score assigned to the response, or None if no scoring was performed
        """

        if not self._objective_scorer or not assistant_responses:
            return None
        
        for piece in assistant_responses:
            scores = await self._objective_scorer.score_async(
                request_response=piece,
                task=objective
            )
            
            # Return first successful score
            for score in scores:
                if score.get_value():
                    return score
            
            # No success - return first score as failure indicator
            if scores:
                return scores[0]
        
        return None