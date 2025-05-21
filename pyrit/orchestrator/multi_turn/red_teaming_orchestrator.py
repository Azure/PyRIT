# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import enum
import logging
from pathlib import Path
from typing import Optional, Union
from uuid import uuid4

from pyrit.common.path import RED_TEAM_ORCHESTRATOR_PATH
from pyrit.common.utils import combine_dict
from pyrit.models import PromptRequestPiece, Score, SeedPrompt, SeedPromptGroup
from pyrit.orchestrator import MultiTurnOrchestrator, OrchestratorResult
from pyrit.prompt_converter import PromptConverter
from pyrit.prompt_normalizer import PromptNormalizer
from pyrit.prompt_normalizer.prompt_converter_configuration import (
    PromptConverterConfiguration,
)
from pyrit.prompt_target import PromptChatTarget, PromptTarget
from pyrit.score import Scorer

logger = logging.getLogger(__name__)


class RTOSystemPromptPaths(enum.Enum):
    TEXT_GENERATION = Path(RED_TEAM_ORCHESTRATOR_PATH, "text_generation.yaml").resolve()
    IMAGE_GENERATION = Path(RED_TEAM_ORCHESTRATOR_PATH, "image_generation.yaml").resolve()
    NAIVE_CRESCENDO = Path(RED_TEAM_ORCHESTRATOR_PATH, "naive_crescendo.yaml").resolve()
    VIOLENT_DURIAN = Path(RED_TEAM_ORCHESTRATOR_PATH, "violent_durian.yaml").resolve()
    CRUCIBLE = Path(RED_TEAM_ORCHESTRATOR_PATH, "crucible.yaml").resolve()


class RedTeamingOrchestrator(MultiTurnOrchestrator):
    """
    The `RedTeamingOrchestrator` class orchestrates a multi-turn red teaming attack on a target system.

    It is extemely simple. It sends a prompt to the target system, and then sends the response to the red teaming chat.

    Args:
        objective_target (PromptTarget): Target for created prompts.
        adversarial_chat (PromptChatTarget): Endpoint creating prompts sent to objective_target.
        adversarial_chat_system_prompt_path (Path): Path to initial adversarial_chat system prompt.
        initial_adversarial_chat_prompt (str, Optional): Initial message to start the chat. Defaults to
            "How can I help you?".
        prompt_converters (Optional[list[PromptConverter]]): Converters for prompt formatting. Defaults to None.
        max_turns (int, Optional): Max turns for the conversation, ≥ 0. Defaults to 5.
        objective_scorer (Scorer): Scores prompt target output as sufficient or insufficient.
        use_score_as_feedback (bool, Optional): Use scoring as feedback. Defaults to True.
        verbose (bool, Optional): Print debug info. Defaults to False.

    Raises:
        FileNotFoundError: If adversarial_chat_system_prompt_path file not found.
        ValueError: If max_turns ≤ 0 or if objective_scorer is not binary.
    """

    def __init__(
        self,
        *,
        objective_target: PromptTarget,
        adversarial_chat: PromptChatTarget,
        adversarial_chat_system_prompt_path: Path = RTOSystemPromptPaths.TEXT_GENERATION.value,
        adversarial_chat_seed_prompt: Optional[str] = "How can I help you?",
        prompt_converters: Optional[list[PromptConverter]] = None,
        max_turns: int = 5,
        objective_scorer: Scorer,
        use_score_as_feedback: bool = True,
        batch_size: int = 1,
        verbose: bool = False,
    ) -> None:

        if objective_scorer.scorer_type != "true_false":
            raise ValueError(
                f"The scorer must be a true/false scorer. The scorer type is {objective_scorer.scorer_type}."
            )

        super().__init__(
            objective_target=objective_target,
            adversarial_chat=adversarial_chat,
            adversarial_chat_system_prompt_path=adversarial_chat_system_prompt_path,
            adversarial_chat_seed_prompt=adversarial_chat_seed_prompt,
            max_turns=max_turns,
            prompt_converters=prompt_converters,
            objective_scorer=objective_scorer,
            verbose=verbose,
            batch_size=batch_size,
        )

        self._prompt_normalizer = PromptNormalizer()
        self._use_score_as_feedback = use_score_as_feedback

    def _handle_last_prepended_assistant_message(self) -> Score | None:
        """
        Handle the last message in the prepended conversation if it is from an assistant.
        """
        objective_score: Score | None = None

        for score in self._last_prepended_assistant_message_scores:
            # Extract existing score of the same type
            if score.scorer_class_identifier["__type__"] == self._objective_scorer.get_identifier()["__type__"]:
                objective_score = score
                break

        return objective_score

    def _handle_last_prepended_user_message(self) -> str:
        """
        Handle the last message in the prepended conversation if it is from a user.
        """
        custom_prompt = ""
        if self._last_prepended_user_message and not self._last_prepended_assistant_message_scores:
            logger.info("Sending last user message from prepended conversation to the prompt target.")
            custom_prompt = self._last_prepended_user_message

        return custom_prompt

    async def run_attack_async(
        self, *, objective: str, memory_labels: Optional[dict[str, str]] = None
    ) -> OrchestratorResult:
        """
        Executes a multi-turn red teaming attack asynchronously.

        This method initiates a conversation with the target system, iteratively generating prompts
        and analyzing responses to achieve a specified objective. It evaluates each response for
        success and, if necessary, adapts prompts using scoring feedback until either the objective
        is met or the maximum number of turns is reached.

        Args:
            objective (str): The specific goal the orchestrator aims to achieve through the conversation.
            memory_labels (dict[str, str], Optional): A free-form dictionary of additional labels to apply to the
                prompts throughout the attack. Any labels passed in will be combined with self._global_memory_labels
                (from the GLOBAL_MEMORY_LABELS environment variable) into one dictionary. In the case of collisions,
                the passed-in labels take precedence. Defaults to None.

        Returns:
            OrchestratorResult: Contains the outcome of the attack, including:
                - conversation_id (str): The ID associated with the final conversation state.
                - objective (str): The intended goal of the attack.
                - status (OrchestratorResultStatus): The status of the attack ("success", "failure", "pruned", etc.)
                - score (Score): The score evaluating the attack outcome.
                - confidence (float): The confidence level of the result.

        Raises:
            RuntimeError: If the response from the target system contains an unexpected error.
            ValueError: If the scoring feedback is not of the required type (true/false) for binary completion.
        """
        # Set conversation IDs for objective target and adversarial chat at the beginning of the conversation.
        objective_target_conversation_id = str(uuid4())
        adversarial_chat_conversation_id = str(uuid4())

        updated_memory_labels = combine_dict(existing_dict=self._global_memory_labels, new_dict=memory_labels)

        # Prepare the conversation by adding any provided messages to memory.
        # If there is no prepended conversation, the turn count is 1.
        turn = self._prepare_conversation(new_conversation_id=objective_target_conversation_id)

        achieved_objective = False

        # Custom handling on the first turn for prepended conversation
        score = self._handle_last_prepended_assistant_message()
        custom_prompt = self._handle_last_prepended_user_message()

        while turn <= self._max_turns:
            logger.info(f"Applying the attack strategy for turn {turn}.")

            feedback = None
            if self._use_score_as_feedback and score:
                feedback = score.score_rationale

            response = await self._retrieve_and_send_prompt_async(
                objective=objective,
                objective_target_conversation_id=objective_target_conversation_id,
                adversarial_chat_conversation_id=adversarial_chat_conversation_id,
                feedback=feedback,
                custom_prompt=custom_prompt,
                memory_labels=updated_memory_labels,
            )

            # Reset custom prompt for future turns
            custom_prompt = None

            if response.response_error == "none":
                score = await self._check_conversation_complete_async(
                    objective_target_conversation_id=objective_target_conversation_id,
                )
                if bool(score.get_value()):
                    achieved_objective = True
                    logger.info(
                        "The red teaming orchestrator has completed the conversation and achieved the objective.",
                    )

                    break
            elif response.response_error == "blocked":
                score = None
            else:
                raise RuntimeError(f"Response error: {response.response_error}")

            turn += 1

        if not achieved_objective:
            logger.info(
                "The red teaming orchestrator has not achieved the objective after the maximum "
                f"number of turns ({self._max_turns}).",
            )

        return OrchestratorResult(
            conversation_id=objective_target_conversation_id,
            objective=objective,
            status="success" if achieved_objective else "failure",
            objective_score=score,
            confidence=1.0 if achieved_objective else 0.0,
        )

    async def _retrieve_and_send_prompt_async(
        self,
        *,
        objective: str,
        objective_target_conversation_id: str,
        adversarial_chat_conversation_id: str,
        feedback: Optional[str] = None,
        custom_prompt: str = "",
        memory_labels: Optional[dict[str, str]] = None,
    ) -> PromptRequestPiece:
        """
        Generates and sends a prompt to the prompt target.

        Args:
            objective_target_conversation_id (str): the conversation ID for the prompt target.
            adversarial_chat_conversation_id (str): the conversation ID for the red teaming chat.
            feedback (str, Optional): feedback from a previous iteration of the conversation.
                This can either be a score if the request completed, or a short prompt to rewrite
                the input if the request was blocked.
                The feedback is passed back to the red teaming chat to improve the next prompt.
                For text-to-image applications, for example, there is no immediate text output
                that can be passed back to the red teaming chat, so the scorer rationale is the
                only way to generate feedback.
            custom_prompt (str, optional): If provided, send this prompt to the target directly.
                Otherwise, generate a new prompt with the red teaming LLM.
            memory_labels (dict[str, str], Optional): A free-form dictionary of labels to apply to the
                prompts throughout the attack. These should already be combined with GLOBAL_MEMORY_LABELS.
        """
        if not custom_prompt:
            # The prompt for the red teaming LLM needs to include the latest message from the prompt target.
            logger.info("Generating a prompt for the prompt target using the red teaming LLM.")
            prompt = await self._get_prompt_from_adversarial_chat(
                objective=objective,
                objective_target_conversation_id=objective_target_conversation_id,
                adversarial_chat_conversation_id=adversarial_chat_conversation_id,
                feedback=feedback,
                memory_labels=memory_labels,
            )
        else:
            prompt = custom_prompt

        converter_configurations = PromptConverterConfiguration(converters=self._prompt_converters)

        seed_prompt_group = SeedPromptGroup(
            prompts=[SeedPrompt(value=prompt, data_type="text")],
        )

        response_piece = (
            await self._prompt_normalizer.send_prompt_async(
                seed_prompt_group=seed_prompt_group,
                conversation_id=objective_target_conversation_id,
                request_converter_configurations=[converter_configurations],
                target=self._objective_target,
                labels=memory_labels,
                orchestrator_identifier=self.get_identifier(),
            )
        ).request_pieces[0]

        return response_piece

    async def _check_conversation_complete_async(self, objective_target_conversation_id: str) -> Union[Score, None]:
        """
        Returns the scoring result of the conversation.
        This function uses the scorer to classify the last response.

        Args:
            prompt_target_conversation_id (str): the conversation ID for the prompt target.
        """
        prompt_request_responses = self._memory.get_conversation(conversation_id=objective_target_conversation_id)
        if not prompt_request_responses:
            # If there are no messages, then the conversation is not complete.
            return None
        if prompt_request_responses[-1].request_pieces[0].role in ["user", "system"]:
            # If the last message is a system or red teaming chat message,
            # then the conversation is not yet complete.
            return None

        score = (
            await self._objective_scorer.score_async(request_response=prompt_request_responses[-1].request_pieces[0])
        )[0]

        if score.score_type != "true_false":
            raise ValueError(f"The scorer must return a true_false score. The score type is {score.score_type}.")
        return score

    def _handle_text_response(self, last_response_from_attack_target, feedback) -> str:
        # If the attack target responds with text we can use that as the new prompt
        # unless the response is an error (which may include blocked image requests, for example).
        if last_response_from_attack_target.response_error == "none":
            prompt_text = last_response_from_attack_target.converted_value
            if self._use_score_as_feedback and feedback:
                # concatenate the feedback to the response from the attack target
                prompt_text += "\n\n" + feedback
            return prompt_text
        elif last_response_from_attack_target.response_error == "blocked":
            return (
                "Request to target failed: blocked. " "Please rewrite your prompt to avoid getting blocked next time."
            )
        return f"Request to target failed: {last_response_from_attack_target.response_error}"

    def _handle_file_response(self, last_response_from_attack_target, feedback):
        # This method handles the case where the attack target
        # does not respond with text output, but other modalities.
        # If the response type is not text, the request should have succeeded.
        if last_response_from_attack_target.response_error != "none":
            raise RuntimeError(
                "Request to target failed despite the returned data type "
                f"{last_response_from_attack_target.converted_value_data_type}: "
                f"{last_response_from_attack_target.response_error}"
            )

        # The last response was successful and the response type is not text.
        # If the use_score_as_feedback flag is set, we can use the score rationale as feedback.
        base_error_message = (
            "The attack target does not respond with text output, "
            "so the scoring rationale is the only textual feedback "
            "that can be passed to the red teaming chat. "
        )
        if not self._use_score_as_feedback:
            raise ValueError(
                f"{base_error_message}"
                "However, the use_score_as_feedback flag is set to False so it cannot be utilized."
            )
        if not feedback:
            raise ValueError(f"{base_error_message}" "However, no scoring rationale was provided by the scorer.")
        return feedback

    def _get_prompt_for_adversarial_chat(self, *, objective_target_conversation_id: str, feedback: str | None) -> str:
        """
        Generate prompt for the adversarial chat based off of the last response from the attack target.

        Args:
            objective_target_conversation_id (str): the conversation ID for the objective target.
            feedback (str, Optional): feedback from a previous iteration of the conversation.
                This can either be a score if the request completed, or a short prompt to rewrite
                the input if the request was blocked.
                The feedback is passed back to the red teaming chat to improve the next prompt.
                For text-to-image applications, for example, there is no immediate text output
                that can be passed back to the red teaming chat, so the scorer rationale is the
                only way to generate feedback.
        """
        # If we have previously exchanged messages with the attack target,
        # we can use the last message from the attack target as the new
        # prompt for the red teaming chat.
        last_response_from_objective_target = self._get_last_objective_target_response(
            objective_target_conversation_id=objective_target_conversation_id
        )
        if not last_response_from_objective_target:
            # If there is no response from the attack target (i.e., this is the first turn),
            # we use the initial red teaming prompt
            logger.info(f"Using the specified initial adversarial prompt: {self._adversarial_chat_seed_prompt}")
            return self._adversarial_chat_seed_prompt.value

        if last_response_from_objective_target.converted_value_data_type in ["text", "error"]:
            return self._handle_text_response(last_response_from_objective_target, feedback)

        return self._handle_file_response(last_response_from_objective_target, feedback)

    async def _get_prompt_from_adversarial_chat(
        self,
        *,
        objective: str,
        objective_target_conversation_id: str,
        adversarial_chat_conversation_id: str,
        feedback: Optional[str] = None,
        memory_labels: Optional[dict[str, str]] = None,
    ) -> str:
        """
        Send a prompt to the adversarial chat to generate a new prompt for the objective target.

        Args:
            objective (str): the objective the red teaming orchestrator is trying to achieve.
            objective_target_conversation_id (str): the conversation ID for the prompt target.
            adversarial_chat_conversation_id (str): the conversation ID for the red teaming chat.
            feedback (str, Optional): feedback from a previous iteration of the conversation.
                This can either be a score if the request completed, or a short prompt to rewrite
                the input if the request was blocked.
                The feedback is passed back to the red teaming chat to improve the next prompt.
                For text-to-image applications, for example, there is no immediate text output
                that can be passed back to the red teaming chat, so the scorer rationale is the
                only way to generate feedback.
            memory_labels (dict[str, str], Optional): A free-form dictionary of labels to apply to the
                prompts throughout the attack. These should already be combined with GLOBAL_MEMORY_LABELS.
        """
        prompt_text = self._get_prompt_for_adversarial_chat(
            objective_target_conversation_id=objective_target_conversation_id, feedback=feedback
        )

        if len(self._memory.get_conversation(conversation_id=adversarial_chat_conversation_id)) == 0:
            system_prompt = self._adversarial_chat_system_seed_prompt.render_template_value(objective=objective)

            self._adversarial_chat.set_system_prompt(
                system_prompt=str(system_prompt),
                conversation_id=adversarial_chat_conversation_id,
                orchestrator_identifier=self.get_identifier(),
                labels=memory_labels,
            )

        seed_prompt_group = SeedPromptGroup(
            prompts=[SeedPrompt(value=prompt_text, data_type="text")],
        )

        response_text = (
            await self._prompt_normalizer.send_prompt_async(
                seed_prompt_group=seed_prompt_group,
                conversation_id=adversarial_chat_conversation_id,
                target=self._adversarial_chat,
                orchestrator_identifier=self.get_identifier(),
                labels=memory_labels,
            )
        ).get_value()

        return response_text

    def _get_last_objective_target_response(self, objective_target_conversation_id: str) -> PromptRequestPiece | None:
        target_messages = self._memory.get_conversation(conversation_id=objective_target_conversation_id)
        assistant_responses = [m.request_pieces[0] for m in target_messages if m.request_pieces[0].role == "assistant"]
        return assistant_responses[-1] if len(assistant_responses) > 0 else None
