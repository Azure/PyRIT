# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import enum
import logging
import pathlib
import uuid
from typing import Optional

from pyrit.attacks.base.attack_config import AttackConverterConfig, AttackScoringConfig
from pyrit.attacks.base.attack_context import SingleTurnAttackContext
from pyrit.attacks.base.attack_result import AttackOutcome, AttackResult
from pyrit.attacks.base.attack_strategy import AttackStrategy
from pyrit.attacks.components.conversation_manager import ConversationManager
from pyrit.common.path import DATASETS_PATH
from pyrit.common.utils import combine_dict
from pyrit.models import (
    PromptRequestResponse,
    Score,
    SeedPrompt,
    SeedPromptDataset,
    SeedPromptGroup,
)
from pyrit.models.literals import ChatMessageRole
from pyrit.prompt_converter import LLMGenericTextConverter
from pyrit.prompt_normalizer import PromptConverterConfiguration, PromptNormalizer
from pyrit.prompt_target import PromptChatTarget, PromptTarget
from pyrit.score import Scorer

logger = logging.getLogger(__name__)


class RolePlayPaths(enum.Enum):
    VIDEO_GAME = pathlib.Path(DATASETS_PATH) / "orchestrators" / "role_play" / "video_game.yaml"
    MOVIE_SCRIPT = pathlib.Path(DATASETS_PATH) / "orchestrators" / "role_play" / "movie_script.yaml"
    TRIVIA_GAME = pathlib.Path(DATASETS_PATH) / "orchestrators" / "role_play" / "trivia_game.yaml"
    PERSUASION_SCRIPT = pathlib.Path(DATASETS_PATH) / "orchestrators" / "role_play" / "persuasion_script.yaml"


class RolePlayAttack(AttackStrategy[SingleTurnAttackContext, AttackResult]):
    """
    Implementation of single-turn role-play attack strategy.

    This class orchestrates a role-play attack where malicious objectives are rephrased
    into role-playing contexts to make them appear more benign and bypass content filters.
    The strategy uses an adversarial chat target to transform the objective into a role-play
    scenario before sending it to the target system.

    The attack flow consists of:
    1. Loading role-play scenarios from a YAML file.
    2. Using an adversarial chat target to rephrase the objective into the role-play context.
    3. Sending the rephrased objective to the target system.
    4. Evaluating the response with scorers if configured.
    5. Retrying on failure up to the configured number of retries.
    6. Returning the attack result

    The strategy supports customization through prepended conversations, converters,
    and multiple scorer types for comprehensive evaluation.
    """

    def __init__(
        self,
        *,
        objective_target: PromptTarget,
        adversarial_chat: PromptChatTarget,
        role_play_definition_path: pathlib.Path,
        attack_converter_config: Optional[AttackConverterConfig] = None,
        attack_scoring_config: Optional[AttackScoringConfig] = None,
        prompt_normalizer: Optional[PromptNormalizer] = None,
        max_attempts_on_failure: int = 0,
    ) -> None:
        """
        Initializes the role-play attack strategy.

        Args:
            objective_target (PromptTarget): The target system to attack.
            adversarial_chat (PromptChatTarget): The adversarial chat target used to rephrase
                objectives into role-play scenarios.
            role_play_definition_path (pathlib.Path): Path to the YAML file containing role-play
                definitions (rephrase instructions, user start turn, assistant start turn).
            attack_converter_config (Optional[AttackConverterConfig]): Configuration for prompt converters.
            attack_scoring_config (Optional[AttackScoringConfig]): Configuration for scoring components.
            prompt_normalizer (Optional[PromptNormalizer]): Normalizer for handling prompts.
            max_attempts_on_failure (int): Maximum number of attempts to retry the attack

        Raises:
            ValueError: If the objective scorer is not a true/false scorer.
            FileNotFoundError: If the role_play_definition_path does not exist.
        """
        # Initialize base class
        super().__init__(logger=logger, context_type=SingleTurnAttackContext)

        # Store the objective target and adversarial chat
        self._objective_target = objective_target
        self._adversarial_chat = adversarial_chat

        # Load role-play definitions
        role_play_definition = SeedPromptDataset.from_yaml_file(role_play_definition_path)
        self._rephrase_instructions = role_play_definition.prompts[0]
        self._user_start_turn = role_play_definition.prompts[1]
        self._assistant_start_turn = role_play_definition.prompts[2]

        # Initialize the converter configuration
        attack_converter_config = attack_converter_config or AttackConverterConfig()
        self._request_converters = attack_converter_config.request_converters
        self._response_converters = attack_converter_config.response_converters

        # Initialize scoring configuration
        attack_scoring_config = attack_scoring_config or AttackScoringConfig()

        # Check for unused optional parameters and warn if they are set
        self._warn_if_set(config=attack_scoring_config, unused_fields=["refusal_scorer"])

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
        if not context.objective:
            raise ValueError("Attack objective must be provided in the context")

        if not context.conversation_id:
            raise ValueError("Conversation ID must be provided in the context")

    async def _setup_async(self, *, context: SingleTurnAttackContext) -> None:
        """
        Set up the attack by preparing conversation context with role-play start.

        Args:
            context (SingleTurnAttackContext): The attack context containing attack parameters.
        """
        # Ensure the context has a conversation ID
        context.conversation_id = str(uuid.uuid4())

        # Combine memory labels from context and attack strategy
        context.memory_labels = combine_dict(self._memory_labels, context.memory_labels)

        # Get role-play conversation start and combine with any existing prepended conversation
        role_play_start = await self._get_conversation_start()
        if role_play_start:
            combined_prepended_conversation = (context.prepended_conversation or []) + role_play_start
        else:
            combined_prepended_conversation = context.prepended_conversation

        # Process prepended conversation including role-play context
        await self._conversation_manager.update_conversation_state_async(
            conversation_id=context.conversation_id,
            prepended_conversation=combined_prepended_conversation,
            converter_configurations=self._request_converters,
        )

    async def _perform_attack_async(self, *, context: SingleTurnAttackContext) -> AttackResult:
        """
        Perform the role-play attack.

        Args:
            context: The attack context with objective and parameters.

        Returns:
            AttackResult containing the outcome of the attack.
        """

        self._logger.info(f"Starting role-play attack with objective: {context.objective}")
        self._logger.info(f"Max attempts: {self._max_attempts_on_failure}")

        # Execute with retries
        response = None
        score = None

        # Role-play attack execution steps:
        # 1) Set up the role-play conversation context using input role-play definitions
        # 2) Send the role-play scenario to adversarial chat target to rephrase the objective
        # 3) Send the rephrased objective (from the adversarial target) to the target model
        # 4) If the call fails or the response is filtered, retry as configured
        # 5) When a response is obtained, optionally evaluate it with the objective scorer
        # 6) Stop early if the objective is achieved; otherwise continue retry loop
        # 7) After retries are exhausted, compile the final response and score
        # 8) Return an AttackResult object that captures the outcome of the attack

        # Prepare the prompt group with the original objective
        prompt_group = self._get_prompt_group(context)

        # Execute with retries
        for attempt in range(self._max_attempts_on_failure + 1):
            self._logger.debug(f"Attempt {attempt+1}/{self._max_attempts_on_failure + 1}")

            # Send the prompt with role-play context (conversation start already set up in _setup_async)
            response = await self._send_prompt_to_objective_target_async(prompt_group=prompt_group, context=context)
            if not response:
                self._logger.warning(f"No response received on attempt {attempt+1} (likely filtered)")
                continue  # Retry if no response (filtered or error)

            # If no objective scorer, we have a response but can't determine success
            if not self._objective_scorer:
                break

            # Score the response
            score = await self._evaluate_response_async(response=response, objective=context.objective)

            # On success, return immediately
            if bool(score and score.get_value()):
                break

        # Determine the outcome
        outcome, outcome_reason = self._determine_attack_outcome(response=response, score=score, context=context)

        result = AttackResult(
            conversation_id=context.conversation_id,
            objective=context.objective,
            attack_identifier=self.get_identifier(),
            last_response=response.get_piece() if response else None,
            last_score=score,
            outcome=outcome,
            outcome_reason=outcome_reason,
            executed_turns=1,
        )

        return result

    async def _teardown_async(self, *, context: SingleTurnAttackContext) -> None:
        """Clean up after attack execution"""
        # Nothing to be done here, no-op
        pass

    async def _get_conversation_start(self) -> Optional[list[PromptRequestResponse]]:
        """
        Get the role-play conversation start messages.

        Returns:
            Optional[list[PromptRequestResponse]]: List containing user and assistant start turns
                for the role-play scenario.
        """
        return [
            PromptRequestResponse.from_prompt(
                prompt=self._user_start_turn.value,
                role="user",
            ),
            PromptRequestResponse.from_prompt(
                prompt=self._assistant_start_turn.value,
                role="assistant",
            ),
        ]

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
        Send the prompt to the target with role-play context and return the response.

        Args:
            prompt_group (SeedPromptGroup): The seed prompt group to send.
            context (SingleTurnAttackContext): The attack context containing parameters and labels.

        Returns:
            Optional[PromptRequestResponse]: The model's response if successful, or None if
                the request was filtered, blocked, or encountered an error.
        """
        # Create rephrase converter configuration that includes the adversarial chat
        rephrase_converter = PromptConverterConfiguration.from_converters(
            converters=[
                LLMGenericTextConverter(
                    converter_target=self._adversarial_chat,
                    user_prompt_template_with_objective=self._rephrase_instructions,
                )
            ]
        )

        # Combine rephrase converter with existing request converters
        combined_converters = rephrase_converter + self._request_converters

        return await self._prompt_normalizer.send_prompt_async(
            seed_prompt_group=prompt_group,
            target=self._objective_target,
            conversation_id=context.conversation_id,
            request_converter_configurations=combined_converters,
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
        role: ChatMessageRole = "assistant"

        # Run auxiliary scorers (no return value needed)
        await Scorer.score_response_async(response=response, scorers=self._auxiliary_scorers, role_filter=role)

        # Run objective scorer
        if self._objective_scorer:
            return await Scorer.score_response_select_first_success_async(
                response=response, scorers=[self._objective_scorer], role_filter=role, task=objective
            )

        return None

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
