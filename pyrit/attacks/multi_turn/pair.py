# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from pyrit.attacks.base.attack_config import (
    AttackAdversarialConfig,
    AttackConverterConfig,
    AttackScoringConfig,
)
from pyrit.attacks.base.attack_context import MultiTurnAttackContext
from pyrit.attacks.base.attack_strategy import AttackStrategy
from pyrit.attacks.multi_turn.tree_of_attacks import TreeOfAttacksWithPruningAttack
from pyrit.common.path import DATASETS_PATH
from pyrit.models import AttackOutcome, AttackResult, SeedPrompt
from pyrit.prompt_normalizer import PromptNormalizer
from pyrit.prompt_target import PromptChatTarget

logger = logging.getLogger(__name__)


@dataclass
class PAIRAttackContext(MultiTurnAttackContext):
    """Context for the PAIR attack strategy."""
    
    # PAIR-specific parameters
    depth: int = 3
    desired_response_prefix: str = "Sure, here is"
    objective_achieved_score_threshold: float = 0.8

    @classmethod
    def create_from_params(
        cls,
        *,
        objective: str,
        prepended_conversation: list,
        memory_labels: dict,
        depth: int = 3,
        desired_response_prefix: str = "Sure, here is",
        objective_achieved_score_threshold: float = 0.8,
        **kwargs
    ) -> "PAIRAttackContext":
        """Create PAIRAttackContext from parameters."""
        return cls(
            objective=objective,
            prepended_conversation=prepended_conversation,
            memory_labels=memory_labels,
            depth=depth,
            desired_response_prefix=desired_response_prefix,
            objective_achieved_score_threshold=objective_achieved_score_threshold,
        )


@dataclass
class PAIRAttackResult(AttackResult):
    """Result of the PAIR attack strategy execution."""
    
    def __post_init__(self):
        """Initialize metadata after dataclass construction."""
        if not hasattr(self, 'metadata') or self.metadata is None:
            self.metadata = {}
    
    @property
    def depth_reached(self) -> int:
        """
        Get the depth reached during the attack.
        
        Returns:
            int: The maximum depth reached.
        """
        return self.metadata.get("depth_reached", 0)
    
    @depth_reached.setter
    def depth_reached(self, value: int) -> None:
        """
        Set the depth reached during the attack.
        
        Args:
            value (int): The depth reached to set.
        """
        self.metadata["depth_reached"] = value


class PAIRAttack(AttackStrategy[PAIRAttackContext, PAIRAttackResult]):
    """
    Implementation of the Prompt Automatic Iterative Refinement (PAIR) attack strategy.

    This algorithm was published and described in the paper: Chao, Patrick, et al. Jailbreaking Black Box Large
    Language Models in Twenty Queries. arXiv:2310.08419, arXiv, 13 Oct. 2023. arXiv.org, http://arxiv.org/abs/2310.08419

    A reference implementation is found in this following repository: 
        https://github.com/patrickrchao/JailbreakingLLMs/blob/main/system_prompts.py

    The PAIR attack uses a tree-based approach with pruning to iteratively refine prompts.
    It leverages an adversarial chat model to generate prompts that attempt to achieve
    the objective against a target system.

    The attack flow consists of:
    1. Generating adversarial prompts using an adversarial chat model.
    2. Sending prompts to the target and evaluating responses.
    3. Pruning unsuccessful branches based on scoring thresholds.
    4. Continuing iteratively until the objective is achieved or maximum depth is reached.
    """

    # Default system prompt template path for PAIR attack
    DEFAULT_ADVERSARIAL_CHAT_SYSTEM_PROMPT_PATH: Path = (
        Path(DATASETS_PATH) / "orchestrators" / "pair" / "attacker_system_prompt.yaml"
    )

    def __init__(
        self,
        *,
        objective_target: PromptChatTarget,
        attack_adversarial_config: AttackAdversarialConfig,
        attack_converter_config: Optional[AttackConverterConfig] = None,
        attack_scoring_config: Optional[AttackScoringConfig] = None,
        prompt_normalizer: Optional[PromptNormalizer] = None,
        depth: int = 3,
        desired_response_prefix: str = "Sure, here is",
        objective_achieved_score_threshold: float = 0.8,
    ) -> None:
        """
        Initialize the PAIR attack strategy.

        Args:
            objective_target (PromptChatTarget): The target system to attack. Must be a PromptChatTarget.
            attack_adversarial_config (AttackAdversarialConfig): Configuration for the adversarial component,
                including the adversarial chat target and optional system prompt path.
            attack_converter_config (Optional[AttackConverterConfig]): Configuration for attack converters,
                including request and response converters.
            attack_scoring_config (Optional[AttackScoringConfig]): Configuration for attack scoring.
            prompt_normalizer (Optional[PromptNormalizer]): The prompt normalizer to use for sending prompts.
            depth (int): The maximum depth of the attack tree. Default is 3.
            desired_response_prefix (str): The desired prefix for responses. Default is "Sure, here is".
            objective_achieved_score_threshold (float): The score threshold to determine if the objective is
                achieved. Default is 0.8.

        Raises:
            ValueError: If objective_target is not a PromptChatTarget.
            ValueError: If depth is less than 1.
            ValueError: If objective_achieved_score_threshold is not between 0 and 1.
        """
        # Initialize base class
        super().__init__(logger=logger, context_type=PAIRAttackContext)

        # Validate inputs
        if not isinstance(objective_target, PromptChatTarget):
            raise ValueError("objective_target must be a PromptChatTarget")

        if depth < 1:
            raise ValueError("The depth of the tree must be at least 1.")

        if objective_achieved_score_threshold < 0 or objective_achieved_score_threshold > 1:
            raise ValueError("The objective achieved score threshold must be between 0 and 1.")

        # Store the objective target
        self._objective_target = objective_target

        # Initialize converter configuration
        attack_converter_config = attack_converter_config or AttackConverterConfig()
        self._request_converters = attack_converter_config.request_converters
        self._response_converters = attack_converter_config.response_converters

        # Initialize scoring configuration
        attack_scoring_config = attack_scoring_config or AttackScoringConfig()
        self._objective_scorer = attack_scoring_config.objective_scorer
        self._auxiliary_scorers = attack_scoring_config.auxiliary_scorers
        self._successful_objective_threshold = attack_scoring_config.successful_objective_threshold

        # Initialize adversarial configuration
        self._adversarial_chat = attack_adversarial_config.target
        system_prompt_path = (
            attack_adversarial_config.system_prompt_path
            or PAIRAttack.DEFAULT_ADVERSARIAL_CHAT_SYSTEM_PROMPT_PATH
        )
        self._adversarial_chat_system_prompt_path = system_prompt_path

        # Initialize PAIR-specific parameters
        self._depth = depth
        self._desired_response_prefix = desired_response_prefix
        self._objective_achieved_score_threshold = objective_achieved_score_threshold

        # Initialize utilities
        self._prompt_normalizer = prompt_normalizer or PromptNormalizer()

        # Initialize the underlying tree attack
        self._tree_attack = None

    def _validate_context(self, *, context: PAIRAttackContext) -> None:
        """
        Validate the PAIR attack context.

        Args:
            context (PAIRAttackContext): The context to validate.

        Raises:
            ValueError: If the context is invalid for this attack strategy.
        """
        if not context.objective:
            raise ValueError("Objective cannot be empty")

        if context.depth < 1:
            raise ValueError("Depth must be at least 1")

        if context.objective_achieved_score_threshold < 0 or context.objective_achieved_score_threshold > 1:
            raise ValueError("Objective achieved score threshold must be between 0 and 1")

    async def _setup_async(self, *, context: PAIRAttackContext) -> None:
        """
        Setup phase for the PAIR attack.

        Args:
            context (PAIRAttackContext): The context for the attack.
        """
        # Create adversarial config
        from pyrit.attacks.base.attack_config import AttackAdversarialConfig, AttackConverterConfig, AttackScoringConfig
        
        adversarial_config = AttackAdversarialConfig(
            target=self._adversarial_chat,
            system_prompt_path=self._adversarial_chat_system_prompt_path,
        )
        
        converter_config = AttackConverterConfig(
            request_converters=self._request_converters,
            response_converters=self._response_converters,
        )
        
        scoring_config = AttackScoringConfig(
            objective_scorer=self._objective_scorer,
            auxiliary_scorers=self._auxiliary_scorers,
            successful_objective_threshold=context.objective_achieved_score_threshold,
        )
        
        # Create the underlying TreeOfAttacksWithPruningAttack with PAIR-specific configuration
        self._tree_attack = TreeOfAttacksWithPruningAttack(
            objective_target=self._objective_target,
            attack_adversarial_config=adversarial_config,
            attack_converter_config=converter_config,
            attack_scoring_config=scoring_config,
            prompt_normalizer=self._prompt_normalizer,
            tree_width=1,  # PAIR uses width=1
            tree_depth=context.depth,
            branching_factor=1,  # PAIR uses branching_factor=1
            on_topic_checking_enabled=False,  # PAIR disables on-topic checking
            desired_response_prefix=context.desired_response_prefix,
            batch_size=1,
        )

    async def _perform_attack_async(self, *, context: PAIRAttackContext) -> PAIRAttackResult:
        """
        Core PAIR attack implementation.

        Args:
            context (PAIRAttackContext): The context for the attack.

        Returns:
            PAIRAttackResult: The result of the attack execution.
        """
        try:
            # Ensure tree_attack was initialized in setup
            if self._tree_attack is None:
                raise ValueError("TreeOfAttacksWithPruningAttack was not initialized properly in setup phase")
                
            # Run the underlying tree attack using the attack strategy interface
            tree_result = await self._tree_attack.execute_async(
                objective=context.objective,
                prepended_conversation=context.prepended_conversation,
                memory_labels=context.memory_labels,
            )

            # Convert the tree result to PAIR result
            if tree_result.outcome == AttackOutcome.SUCCESS:
                outcome = AttackOutcome.SUCCESS
                outcome_reason = "PAIR attack achieved the objective"
            elif tree_result.outcome == AttackOutcome.FAILURE:
                outcome = AttackOutcome.FAILURE
                outcome_reason = "PAIR attack failed to achieve the objective"
            else:
                outcome = AttackOutcome.UNDETERMINED
                outcome_reason = f"PAIR attack completed with outcome: {tree_result.outcome.value}"

            # Create the result
            result = PAIRAttackResult(
                attack_identifier=self.get_identifier(),
                conversation_id=tree_result.conversation_id,
                objective=context.objective,
                last_response=tree_result.last_response,
                last_score=tree_result.last_score,
                executed_turns=tree_result.executed_turns,
                execution_time_ms=tree_result.execution_time_ms,
                outcome=outcome,
                outcome_reason=outcome_reason,
            )

            # Set depth reached
            result.depth_reached = context.depth

            return result

        except Exception as e:
            self._logger.error(f"PAIR attack failed with error: {str(e)}")
            return PAIRAttackResult(
                attack_identifier=self.get_identifier(),
                conversation_id=context.session.conversation_id,
                objective=context.objective,
                outcome=AttackOutcome.FAILURE,
                outcome_reason=f"PAIR attack failed with error: {str(e)}",
                executed_turns=0,
            )

    async def _teardown_async(self, *, context: PAIRAttackContext) -> None:
        """
        Teardown phase for the PAIR attack.

        Args:
            context (PAIRAttackContext): The context for the attack.
        """
        # Clean up any resources if needed
        self._tree_attack = None
        self._logger.debug("PAIR attack teardown completed")