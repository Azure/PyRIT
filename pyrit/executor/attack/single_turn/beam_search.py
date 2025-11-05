# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging
import uuid
from typing import Optional

from pyrit.common.apply_defaults import apply_defaults
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
    Message,
    Score,
    SeedGroup,
    SeedPrompt,
)
from pyrit.prompt_normalizer import PromptNormalizer
from pyrit.prompt_target import PromptTarget
from pyrit.score import Scorer

logger = logging.getLogger(__name__)


class BeamSearchAttack(SingleTurnAttackStrategy):
    @apply_defaults
    def __init__(
        self,
        *,
        objective_target: PromptTarget,
        attack_scoring_config: AttackScoringConfig,
        attack_converter_config: Optional[AttackConverterConfig] = None,
        prompt_normalizer: Optional[PromptNormalizer] = None,
        num_beams: int = 5,
        max_iterations: int = 10,
    ) -> None:
        """
        Initialize the prompt injection attack strategy.

        Args:
            objective_target (PromptTarget): The target system to attack.
            attack_converter_config (Optional[AttackConverterConfig]): Configuration for prompt converters.
            attack_scoring_config (Optional[AttackScoringConfig]): Configuration for scoring components.

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
        assert len(attack_scoring_config.auxiliary_scorers) > 0, "At least one auxiliary scorer must be provided."

        # Check for unused optional parameters and warn if they are set
        warn_if_set(config=attack_scoring_config, unused_fields=["refusal_scorer"], log=logger)

        self._auxiliary_scorers = attack_scoring_config.auxiliary_scorers
        self._objective_scorer = attack_scoring_config.objective_scorer

        # Skip criteria could be set directly in the injected prompt normalizer
        self._prompt_normalizer = prompt_normalizer or PromptNormalizer()
        self._conversation_manager = ConversationManager(
            attack_identifier=self.get_identifier(),
            prompt_normalizer=self._prompt_normalizer,
        )

        self._num_beams = num_beams
        self._max_iterations = max_iterations

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
        Perform the attack.

        Args:
            context: The attack context with objective and parameters.

        Returns:
            AttackResult containing the outcome of the attack.
        """
        # Log the attack configuration
        self._logger.info(f"Starting {self.__class__.__name__} with objective: {context.objective}")

        # Execute with retries
        response = None
        score = None

        # Prepare the prompt
        prompt_group = self._get_prompt_group(context)
        print(f"Prepared prompt group: {prompt_group}")

        
        result = AttackResult(
            conversation_id=context.conversation_id,
            objective=context.objective,
            attack_identifier=self.get_identifier(),
            last_response=response.get_piece() if response else None,
            last_score=score,
            related_conversations=context.related_conversations,
            # outcome=outcome,
            # outcome_reason=outcome_reason,
            executed_turns=1,
        )

        return result
    
    
    def _get_prompt_group(self, context: SingleTurnAttackContext) -> SeedGroup:
        """
        Prepare the seed group for the attack.

        If a seed_group is provided in the context, it will be used directly.
        Otherwise, creates a new SeedGroup with the objective as a text prompt.

        Args:
            context (SingleTurnAttackContext): The attack context containing the objective
                and optionally a pre-configured seed_group.

        Returns:
            SeedGroup: The seed group to be used in the attack.
        """
        if context.seed_group:
            return context.seed_group

        return SeedGroup(prompts=[SeedPrompt(value=context.objective, data_type="text")])
    
    
    async def _teardown_async(self, *, context: SingleTurnAttackContext) -> None:
        """Clean up after attack execution"""
        # Nothing to be done here, no-op
        pass