# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

import enum
import logging
from pathlib import Path
from typing import Optional, Union

from pyrit.attacks.base.attack_strategy import AttackStrategy
from pyrit.attacks.base.context import ConversationSession, MultiTurnAttackContext
from pyrit.attacks.base.result import AttackResult
from pyrit.attacks.components.conversation_manager import (
    ConversationManager,
    ConversationState,
)
from pyrit.attacks.components.score_evaluator import ScoreEvaluator
from pyrit.common.path import RED_TEAM_ORCHESTRATOR_PATH
from pyrit.common.utils import combine_dict
from pyrit.models import SeedPrompt
from pyrit.models.prompt_request_piece import PromptRequestPiece
from pyrit.models.score import Score
from pyrit.models.seed_prompt import SeedPromptGroup
from pyrit.prompt_normalizer.prompt_converter_configuration import (
    PromptConverterConfiguration,
)
from pyrit.prompt_normalizer.prompt_normalizer import PromptNormalizer
from pyrit.prompt_target.common.prompt_chat_target import PromptChatTarget
from pyrit.prompt_target.common.prompt_target import PromptTarget
from pyrit.score.scorer import Scorer

logger = logging.getLogger(__name__)

DEFAULT_ADVERSARIAL_PROMPT_IF_OBJECTIVE_TARGET_IS_BLOCKED = (
    "Request to target failed: blocked. Please rewrite your prompt to avoid getting blocked next time."
)

DEFAULT_ERR_MSG_IF_OBJECTIVE_TARGET_HAS_NON_TEXT_RESPONSE = (
    "The attack target does not respond with text output, "
    "so the scoring rationale is the only textual feedback "
    "that can be passed to the red teaming chat. "
)


class RTOSystemPromptPaths(enum.Enum):
    TEXT_GENERATION = Path(RED_TEAM_ORCHESTRATOR_PATH, "text_generation.yaml").resolve()
    IMAGE_GENERATION = Path(RED_TEAM_ORCHESTRATOR_PATH, "image_generation.yaml").resolve()
    NAIVE_CRESCENDO = Path(RED_TEAM_ORCHESTRATOR_PATH, "naive_crescendo.yaml").resolve()
    VIOLENT_DURIAN = Path(RED_TEAM_ORCHESTRATOR_PATH, "violent_durian.yaml").resolve()
    CRUCIBLE = Path(RED_TEAM_ORCHESTRATOR_PATH, "crucible.yaml").resolve()


class RedTeamingAttack(AttackStrategy[MultiTurnAttackContext, AttackResult]):
    """
    Implementation of multi-turn red teaming attack strategy.

    This class orchestrates an iterative attack process where an adversarial chat model generates
    prompts to send to a target system, attempting to achieve a specified objective. The strategy
    evaluates each target response using a scorer to determine if the objective has been met.

    The attack flow consists of:
    1. Generating adversarial prompts based on previous responses and scoring feedback
    2. Sending prompts to the target system through optional converters
    3. Scoring target responses to assess objective achievement
    4. Using scoring feedback to guide subsequent prompt generation
    5. Continuing until the objective is achieved or maximum turns are reached

    The strategy supports customization through system prompts, seed prompts, and prompt converters,
    allowing for various attack techniques and scenarios.
    """

    def __init__(
        self,
        *,
        objective_target: PromptTarget,
        objective_scorer: Scorer,
        adversarial_chat: PromptChatTarget,
        adversarial_chat_system_prompt_path: Union[str, Path] = RTOSystemPromptPaths.TEXT_GENERATION.value,
        adversarial_chat_seed_prompt: Union[str, SeedPrompt] = "How can I help you?",
        prompt_normalizer: Optional[PromptNormalizer] = None,
        use_score_as_feedback: bool = False,
    ):
        """
        Initialize the red teaming attack strategy

        Args:
            objective_target (PromptTarget): The target system to attack
            objective_scorer (Scorer): The scorer to evaluate the target's responses
            adversarial_chat (PromptChatTarget): The adversarial chat model to use for generating prompts
            adversarial_chat_system_prompt_path (Union[str, Path]): Path to the YAML file containing
                                                                    the system prompt template
            adversarial_chat_seed_prompt (Union[str, SeedPrompt]): The seed prompt for the adversarial chat
            prompt_normalizer (Optional[PromptNormalizer]): The prompt normalizer to use for sending prompts
            use_score_as_feedback (bool): Whether to use the score as feedback for generating prompts
        """
        super().__init__(
            logger=logger,
        )

        self._objective_target = objective_target
        self._objective_scorer = objective_scorer
        self._adversarial_chat = adversarial_chat
        self._set_adversarial_chat_system_prompt_template(
            system_prompt_template_path=adversarial_chat_system_prompt_path
        )
        self._set_adversarial_chat_seed_prompt(seed_prompt=adversarial_chat_seed_prompt)
        self._prompt_normalizer = prompt_normalizer or PromptNormalizer()
        self._use_score_as_feedback = use_score_as_feedback
        self._conversation_manager = ConversationManager(attack_identifier=self.get_identifier())
        self._score_evaluator = ScoreEvaluator(
            use_score_as_feedback=use_score_as_feedback,
            scorer=objective_scorer,
        )

    def _validate_context(self, *, context: MultiTurnAttackContext) -> None:
        """
        Validate the context before executing the attack

        Args:
            context (MultiTurnAttackContext): The context to validate

        Raises:
            ValueError: If the context is invalid
        """
        if not context.objective:
            raise ValueError("Attack objective must be provided")

        if context.max_turns <= 0:
            raise ValueError("Max turns must be greater than 0")

    async def _setup_async(self, *, context: MultiTurnAttackContext) -> None:
        """
        Prepare the strategy for execution

        1. Initializes or retrieves the conversation state
        2. Updates turn counts and checks for any custom prompt
        3. Retrieves the last assistant message's evaluation score if available
        4. Merges memory labels from context
        5. Sets the system prompt for adversarial chat

        Args:
            context (MultiTurnAttackContext): Attack context with configuration

        Raises:
            ValueError: If the system prompt is not defined
        """
        # Initialize the conversation session if not already set
        context.session = context.session or ConversationSession()
        logger.debug(f"Conversation session ID: {context.session.conversation_id}")
        logger.debug(f"Adversarial chat conversation ID: {context.session.adversarial_chat_conversation_id}")

        # Explicitly set achieved objective to False
        context.achieved_objective = False

        # Set up prompt converters if provided
        converter_configurations = (
            [PromptConverterConfiguration(converters=context.prompt_converters)] if context.prompt_converters else []
        )

        # Update the conversation state with the current context
        conversation_state: ConversationState = await self._conversation_manager.update_conversation_state_async(
            target=self._objective_target,
            max_turns=context.max_turns,
            conversation_id=context.session.conversation_id,
            prepended_conversation=context.prepended_conversation,
            converter_configurations=converter_configurations,
        )

        # update the turns based on prepend conversation
        context.executed_turns = conversation_state.turn_count

        # update the custom prompt if provided
        if RedTeamingAttack._has_custom_prompt(state=conversation_state):
            context.custom_prompt = conversation_state.last_user_message

        # get the last assistant message evaluation score if available
        score = self._retrieve_last_assistant_message_evaluation_score(state=conversation_state)
        context.last_score = score

        # update the memory labels
        context.memory_labels = combine_dict(existing_dict=self._memory_labels, new_dict=context.memory_labels or {})

        # set the system prompt for the adversarial chat
        system_prompt = self._adversarial_chat_system_prompt_template.render_template_value(objective=context.objective)
        if not system_prompt:
            raise ValueError("Adversarial chat system prompt must be defined")

        self._adversarial_chat.set_system_prompt(
            system_prompt=system_prompt,
            conversation_id=context.session.adversarial_chat_conversation_id,
            orchestrator_identifier=self.get_identifier(),
            labels=context.memory_labels,
        )

    async def _perform_attack_async(self, *, context: MultiTurnAttackContext) -> AttackResult:
        """
        Execute the red teaming attack by iteratively generating prompts,
        sending them to the target, and scoring the responses in a loop
        until the objective is achieved or the maximum turns are reached

        Args:
            context (MultiTurnAttackContext): The attack context containing configuration and state

        Returns:
            AttackResult: The result of the attack execution
        """

        # Log the attack configuration
        logger.info(f"Starting red teaming attack with objective: {context.objective}")
        logger.info(f"Max turns: {context.max_turns}")

        # Attack Execution Steps:
        # 1) Generate adversarial prompt based on previous feedback or custom prompt
        # 2) Send the generated prompt to the target system
        # 3) Evaluate the target's response using the objective scorer
        # 4) Check if the attack objective has been achieved
        # 5) Repeat steps 1-4 until objective is achieved or max turns are reached

        # Execute conversation turns
        while context.executed_turns < context.max_turns and not context.achieved_objective:
            logger.info(f"Executing turn {context.executed_turns + 1}/{context.max_turns}")

            # Determine what to send next
            prompt_to_send = await self._generate_next_prompt(context=context)

            # Send the generated prompt to target
            target_response = await self._send_prompt_to_target(context=context, prompt=prompt_to_send)
            context.last_response = target_response

            # Score the response
            score = await self._score_response(context=context, response=target_response)
            context.last_score = score

            # Check if objective achieved
            context.achieved_objective = self._score_evaluator.is_objective_achieved(score=score)

            # Increment the executed turns
            context.executed_turns += 1

        # Log the result of the attack
        self._log_objective_status(context)

        # Prepare the result
        return AttackResult(
            orchestrator_identifier=self.get_identifier(),
            conversation_id=context.session.conversation_id,
            objective=context.objective,
            achieved_objective=context.achieved_objective,
            executed_turns=context.executed_turns,
            last_response=context.last_response,
            last_score=context.last_score,
        )

    async def _teardown_async(self, *, context: MultiTurnAttackContext) -> None:
        """Clean up after attack execution"""
        # Nothing to be done here, no-op
        pass

    async def _generate_next_prompt(self, context: MultiTurnAttackContext) -> str:
        """
        Generate the next prompt to be sent to the target during the red teaming attack

        This method is called each turn to obtain fresh adversarial text based on previous feedback,
        error states, or the custom prompt if it is the first turn. It integrates feedback from the
        scorer when available, and handles blocked or error responses by returning fallback prompts

        Args:
            context (MultiTurnAttackContext): The attack context containing the current state and configuration

        Returns:
            str: The generated prompt to be sent to the adversarial chat
        """
        # If first turn and custom prompt provided, use it
        if context.executed_turns == 0 and context.custom_prompt:
            logger.debug("Using custom prompt for first turn")
            return context.custom_prompt

        # Generate prompt using adversarial chat
        logger.debug(f"Generating prompt for turn {context.executed_turns + 1}")

        # Prepare prompt for the adversarial chat
        prompt_text = await self._build_adversarial_prompt(context)

        # Send the prompt to the adversarial chat and get the response
        logger.debug(f"Sending prompt to adversarial chat: {prompt_text[:50]}...")
        prompt_grp = SeedPromptGroup(prompts=[SeedPrompt(value=prompt_text, data_type="text")])

        response = await self._prompt_normalizer.send_prompt_async(
            seed_prompt_group=prompt_grp,
            conversation_id=context.session.adversarial_chat_conversation_id,
            target=self._adversarial_chat,
            orchestrator_identifier=self.get_identifier(),
            labels=context.memory_labels,
        )

        # Check if the response is valid
        if response is None:
            raise ValueError("Received no response from adversarial chat")

        return response.get_value()

    async def _build_adversarial_prompt(
        self,
        context: MultiTurnAttackContext,
    ) -> str:
        """
        Build a prompt for the adversarial chat

        Args:
            context (MultiTurnAttackContext): The attack context containing the current state and configuration

        Returns:
            str: The prompt to be sent to the adversarial chat
        """
        # Get the last assistant message from the conversation manager
        response = self._conversation_manager.get_last_message(
            conversation_id=context.session.conversation_id, role="assistant"
        )
        if not response:
            return self._adversarial_chat_seed_prompt.value

        # Build the prompt based on the last response from the target
        if response.converted_value_data_type in ["text", "error"]:
            return self._handle_adversarial_text_response(response=response, context=context)
        return self._handle_adversarial_file_response(response=response, context=context)

    def _handle_adversarial_text_response(
        self, *, response: PromptRequestPiece, context: MultiTurnAttackContext
    ) -> str:
        """
        Handle the text response from the target by appending any
        available scoring feedback to the returned text. If the response
        indicates a block or error, return a fallback message instead

        Args:
            response (PromptRequestPiece): The response from the target
            context (MultiTurnAttackContext): The attack context

        Returns:
            str: The text to be sent to the adversarial chat in the next turn
        """
        feedback = self._score_evaluator.get_feedback(context.last_score) if context.last_score else None
        if not response.has_error():
            # if response has no error, we can use the converted value
            prompt_text = response.converted_value
            if not prompt_text:
                logger.warning("Received no converted_value from response")
                return "The previous response was empty. Please continue."

            # if we have feedback, append it to the prompt
            # to provide more context to the adversarial chat
            if feedback:
                prompt_text += f"\n\n{feedback}"
            return prompt_text

        elif response.is_blocked():
            return DEFAULT_ADVERSARIAL_PROMPT_IF_OBJECTIVE_TARGET_IS_BLOCKED

        return f"Request to target failed: {response.response_error}"

    def _handle_adversarial_file_response(
        self, *, response: PromptRequestPiece, context: MultiTurnAttackContext
    ) -> str:
        """
        Handle the file response from the target

        If the response indicates an error, raise a RuntimeError. When scoring is disabled or no
        scoring rationale is provided, raise a ValueError. Otherwise, return the textual feedback as the prompt

        Args:
            response (PromptRequestPiece): The response from the target containing file or non-text data.
            context (MultiTurnAttackContext): The attack context

        Returns:
            str: The suitable feedback or error message to pass back to the adversarial chat.
        """
        if response.has_error():
            raise RuntimeError(
                "Request to target failed despite the returned data type "
                f"{response.converted_value_data_type}: "
                f"{response.response_error}"
            )

        if not self._use_score_as_feedback:
            raise ValueError(
                f"{DEFAULT_ERR_MSG_IF_OBJECTIVE_TARGET_HAS_NON_TEXT_RESPONSE}"
                "However, the use_score_as_feedback flag is set to False so it cannot be utilized."
            )

        feedback = self._score_evaluator.get_feedback(context.last_score) if context.last_score else None
        if not feedback:
            raise ValueError(
                f"{DEFAULT_ERR_MSG_IF_OBJECTIVE_TARGET_HAS_NON_TEXT_RESPONSE}"
                "However, no scoring rationale was provided by the scorer."
            )

        return feedback

    async def _send_prompt_to_target(self, *, context: MultiTurnAttackContext, prompt: str) -> PromptRequestPiece:
        """
        Send a prompt to the target system

        Constructs a seed prompt group, sends it to the target via the prompt normalizer,
        and returns the response as a PromptRequestPiece.

        Args:
            context (MultiTurnAttackContext): The current attack context
            prompt (str): The prompt to send to the target

        Returns:
            PromptRequestPiece: The system's response to the prompt
        """

        converter_cfgs = (
            [PromptConverterConfiguration(converters=context.prompt_converters)] if context.prompt_converters else []
        )

        logger.info(f"Sending prompt to target: {prompt[:50]}...")

        # Create a seed prompt group from the prompt
        seed_prompt = SeedPrompt(value=prompt, data_type="text")
        seed_prompt_group = SeedPromptGroup(prompts=[seed_prompt])

        # Send the prompt to the target
        response = await self._prompt_normalizer.send_prompt_async(
            seed_prompt_group=seed_prompt_group,
            conversation_id=context.session.conversation_id,
            request_converter_configurations=converter_cfgs,
            target=self._objective_target,
            labels=context.memory_labels,
            orchestrator_identifier=self.get_identifier(),
        )

        if response is None:
            # Easiest way to handle this is to raise an error
            # since we cannot continue without a response
            # A proper way to handle this would be to either retry or mark the return as Optional and return None
            # but this would require a lot of changes in the code
            raise ValueError(
                "Received no response from the target system. "
                "Please check the target configuration and ensure it is reachable."
            )

        return response.get_piece()

    async def _score_response(
        self, *, context: MultiTurnAttackContext, response: PromptRequestPiece
    ) -> Optional[Score]:
        """
        Evaluate the target's response with the objective scorer

        Checks if the response is blocked or has an error before scoring
        Returns the resulting Score object or None if the response was blocked

        Args:
            response (PromptRequestPiece): The target system's response
            context (MultiTurnAttackContext): The attack context

        Returns:
            Optional[Score]: The score of the response if available, otherwise None
        """
        if not response.has_error():
            scores = await self._objective_scorer.score_async(
                request_response=response,
                task=context.objective,
            )
            return scores[0] if scores else None
        elif response.is_blocked():
            return None

        raise RuntimeError(f"Response error: {response.response_error}")

    def _log_objective_status(self, context: MultiTurnAttackContext) -> None:
        """
        Log the status of the objective after the attack execution
        This method logs whether the objective was achieved and the number of turns taken

        Args:
            context (MultiTurnAttackContext): The attack context containing execution details
        """
        if context.achieved_objective:
            logger.info(f"Objective achieved on turn {context.executed_turns}")
        else:
            logger.info(
                "The red teaming attack has not achieved the objective after the maximum "
                f"number of turns ({context.max_turns}).",
            )

    @staticmethod
    def _has_custom_prompt(state: ConversationState) -> bool:
        """
        Check if the last user message is considered a custom prompt

        A custom prompt is assumed if the user message exists and no assistant
        message scores are present, indicating a fresh prompt not yet evaluated

        Args:
            state (ConversationState): The conversation state

        Returns:
            bool: True if the last user message is a custom prompt; otherwise, False
        """
        return bool(state.last_user_message and not state.last_assistant_message_scores)

    def _retrieve_last_assistant_message_evaluation_score(self, state: ConversationState) -> Optional[Score]:
        """
        Retrieve the last assistant message evaluation score

        Searches through the last assistant message scores to find one that matches
        the objective scorer type (based on the scorer class identifier).

        Args:
            state (ConversationState): The conversation state

        Returns:
            Optional[Score]: The score of the last assistant message that matches
                           the objective scorer type, or None if not found
        """
        if not state.last_assistant_message_scores:
            return None

        objective_score: Optional[Score] = None
        # Find the score that matches the objective scorer type
        # This is necessary to ensure we are using the correct score for evaluation
        for score in state.last_assistant_message_scores:
            if score.scorer_class_identifier["__type__"] == self._score_evaluator.scorer_type:
                objective_score = score
                break

        return objective_score

    def _set_adversarial_chat_seed_prompt(self, *, seed_prompt: Union[str, SeedPrompt]) -> None:
        """
        Set the seed prompt for the adversarial chat

        Args:
            seed_prompt (Union[str, SeedPrompt]): The seed prompt to set for the adversarial chat

        Raises:
            ValueError: If the seed prompt is not a string or SeedPrompt object
        """
        if isinstance(seed_prompt, str):
            self._adversarial_chat_seed_prompt = SeedPrompt(value=seed_prompt, data_type="text")
        elif isinstance(seed_prompt, SeedPrompt):
            self._adversarial_chat_seed_prompt = seed_prompt
        else:
            raise ValueError("Seed prompt must be a string or SeedPrompt object.")

    def _set_adversarial_chat_system_prompt_template(self, *, system_prompt_template_path: Union[str, Path]) -> None:
        """
        Set the system prompt template for the adversarial chat
        This method loads the system prompt template from a YAML file and checks if it contains an objective

        Args:
            system_prompt_template_path (Union[str, Path]): Path to the YAML file containing the system prompt template

        Raises:
            ValueError: If the system prompt template does not contain an objective
        """
        # Load the system prompt template from the specified YAML file
        sp = SeedPrompt.from_yaml_file(system_prompt_template_path)
        if sp.parameters is None or "objective" not in sp.parameters:
            raise ValueError(f"Adversarial seed prompt must have an objective: '{sp}'")
        self._adversarial_chat_system_prompt_template = sp
