# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import enum
import logging
from pathlib import Path
from typing import Optional, Union
from uuid import uuid4

from pyrit.common.path import RED_TEAM_ORCHESTRATOR_PATH
from pyrit.memory import MemoryInterface
from pyrit.models import SystemPromptWithObjective, PromptRequestPiece
from pyrit.orchestrator import MultiTurnOrchestrator, MultiTurnAttackResult
from pyrit.prompt_normalizer import NormalizerRequestPiece, PromptNormalizer, NormalizerRequest
from pyrit.prompt_target import PromptTarget, PromptChatTarget
from pyrit.prompt_converter import PromptConverter
from pyrit.score import Scorer, Score

logger = logging.getLogger(__name__)

class RTOSystemPromptPaths(enum.Enum):
    TEXT_GENERATION = Path(RED_TEAM_ORCHESTRATOR_PATH, "text_generation.yaml").resolve()
    IMAGE_GENERATION = Path(RED_TEAM_ORCHESTRATOR_PATH, "image_generation.yaml").resolve()
    NAIVE_CRESCENDO = Path(RED_TEAM_ORCHESTRATOR_PATH, "naive_crescendo.yaml").resolve()
    VIOLENT_DURIAN = Path(RED_TEAM_ORCHESTRATOR_PATH, "violent_durian.yaml").resolve()
    CRUCIBLE = Path(RED_TEAM_ORCHESTRATOR_PATH, "crucible.yaml").resolve()


class RedTeamingOrchestrator(MultiTurnOrchestrator):
    _memory: MemoryInterface

    def __init__(
        self,
        *,
        prompt_target: PromptTarget,
        red_team_target: PromptChatTarget,
        red_team_target_system_prompt_path: Path = RTOSystemPromptPaths.TEXT_GENERATION.value,
        initial_red_teaming_prompt: Optional[str] = "How can I help you?",
        prompt_converters: Optional[list[PromptConverter]] = None,
        max_turns: int = 5,
        objective_scorer: Scorer,
        use_score_as_feedback: bool = True,
        memory: Optional[MemoryInterface] = None,
        memory_labels: Optional[dict[str, str]] = None,
        verbose: bool = False,
    ) -> None:
        """Creates an orchestrator to manage conversations between a red teaming target and a prompt target.

        Args:
            prompt_target: The target to send the created prompts to.
            red_teaming_chat: The endpoint that creates prompts that are sent to the prompt target.
            system_prompt_path: The yaml file that contains the attack strategy to use for the red teaming orchestrator.
            initial_red_teaming_prompt: The initial prompt to send to the red teaming target.
                The attack_strategy only provides the strategy, but not the starting point of the conversation.
                The initial_red_teaming_prompt is used to start the conversation with the red teaming target.
                The default is a text prompt with the content "Begin Conversation".
            prompt_converters: The prompt converters to use to convert the prompts before sending them to the prompt
                target. The converters are not applied on messages to the red teaming target.
            objective_scorer: The scorer classifies the prompt target outputs as sufficient (True) or insufficient
                (False) to satisfy the objective that is specified in the attack_strategy.
            use_score_as_feedback: Whether to use the score as feedback to the red teaming chat to improve the
                next prompt.
            memory: The memory to use to store the chat messages. If not provided, a DuckDBMemory will be used.
            memory_labels (dict[str, str], optional): A free-form dictionary for tagging prompts with custom labels.
                These labels can be used to track all prompts sent as part of an operation, score prompts based on
                the operation ID (op_id), and tag each prompt with the relevant Responsible AI (RAI) harm category.
                Users can define any key-value pairs according to their needs. Defaults to None.
                verbose: Whether to print debug information.
        """

        super().__init__(
            prompt_target=prompt_target,
            red_team_target=red_team_target,
            red_team_target_system_prompt_path=red_team_target_system_prompt_path,
            max_turns=max_turns,
            prompt_converters=prompt_converters,
            objective_scorer=objective_scorer,
            memory=memory,
            memory_labels=memory_labels,
            verbose=verbose,
        )

        self._prompt_normalizer = PromptNormalizer(memory=self._memory)

        self._initial_red_teaming_prompt = initial_red_teaming_prompt
        self._use_score_as_feedback = use_score_as_feedback

    async def run_attack_async(self, *, objective: str) -> MultiTurnAttackResult:
        """
        Applies the attack strategy until the conversation is complete or the maximum number of turns is reached.

        Args:
            objective: The objective the red teaming orchestrator is trying to achieve.

        Returns:
            prompt_target_conversation_id: The conversation ID for the multi-turn conversation.
        """
        # Set conversation IDs for prompt target and red teaming chat at the beginning of the conversation.
        prompt_target_conversation_id = str(uuid4())
        red_teaming_chat_conversation_id = str(uuid4())

        turn = 1
        achieved_objective = False
        score: Score | None = None
        while turn <= self._max_turns:
            logger.info(f"Applying the attack strategy for turn {turn}.")

            feedback = None
            if self._use_score_as_feedback and score:
                feedback = score.score_rationale

            response = await self._retrieve_and_send_prompt_async(
                objective=objective,
                prompt_target_conversation_id=prompt_target_conversation_id,
                red_teaming_chat_conversation_id=red_teaming_chat_conversation_id,
                feedback=feedback,
            )

            if response.response_error == "none":
                score = await self._check_conversation_complete_async(
                    prompt_target_conversation_id=prompt_target_conversation_id,
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

        return MultiTurnAttackResult(
            conversation_id=prompt_target_conversation_id,
            achieved_objective=achieved_objective,
            objective=objective,
        )

    async def _retrieve_and_send_prompt_async(
        self,
        *,
        objective: str,
        prompt_target_conversation_id: str,
        red_teaming_chat_conversation_id: str,
        feedback: Optional[str] = None,
    ) -> PromptRequestPiece:
        """
        Generates and sends a prompt to the prompt target.

        Args:
            prompt_target_conversation_id (str): the conversation ID for the prompt target.
            red_teaming_chat_conversation_id (str): the conversation ID for the red teaming chat.
            feedback (str, optional): feedback from a previous iteration of the conversation.
                This can either be a score if the request completed, or a short prompt to rewrite
                the input if the request was blocked.
                The feedback is passed back to the red teaming chat to improve the next prompt.
                For text-to-image applications, for example, there is no immediate text output
                that can be passed back to the red teaming chat, so the scorer rationale is the
                only way to generate feedback.
        """
        # The prompt for the red teaming LLM needs to include the latest message from the prompt target.
        logger.info("Generating a prompt for the prompt target using the red teaming LLM.")
        prompt = await self._get_prompt_from_red_teaming_target(
            objective=objective,
            prompt_target_conversation_id=prompt_target_conversation_id,
            red_teaming_chat_conversation_id=red_teaming_chat_conversation_id,
            feedback=feedback,
        )

        target_prompt_obj = NormalizerRequestPiece(
            request_converters=self._prompt_converters,
            prompt_value=prompt,
            prompt_data_type="text",
            memory=self._memory,
        )

        normalizer_request = NormalizerRequest(
            request_pieces=[target_prompt_obj],
            conversation_id=prompt_target_conversation_id,
        )

        response_piece = (
            await self._prompt_normalizer.send_prompt_async(
                normalizer_request=normalizer_request,
                target=self._prompt_target,
                labels=self._global_memory_labels,
                orchestrator_identifier=self.get_identifier(),
            )
        ).request_pieces[0]

        return response_piece

    async def _check_conversation_complete_async(self, prompt_target_conversation_id: str) -> Union[Score, None]:
        """
        Returns the scoring result of the conversation.
        This function uses the scorer to classify the last response.

        Args:
            prompt_target_conversation_id (str): the conversation ID for the prompt target.
        """
        prompt_request_responses = self._memory.get_conversation(conversation_id=prompt_target_conversation_id)
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

    def _get_prompt_for_red_teaming_chat(self, *, prompt_target_conversation_id: str, feedback: str | None) -> str:
        """
        Generate prompt for the red teaming chat based off of the last response from the attack target.

        Args:
            prompt_target_conversation_id (str): the conversation ID for the prompt target.
            feedback (str, optional): feedback from a previous iteration of the conversation.
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
        last_response_from_attack_target = self._get_last_attack_target_response(
            prompt_target_conversation_id=prompt_target_conversation_id
        )
        if not last_response_from_attack_target:
            # If there is no response from the attack target (i.e., this is the first turn),
            # we use the initial red teaming prompt
            logger.info(f"Using the specified initial red teaming prompt: {self._initial_red_teaming_prompt}")
            return self._initial_red_teaming_prompt

        if last_response_from_attack_target.converted_value_data_type in ["text", "error"]:
            return self._handle_text_response(last_response_from_attack_target, feedback)

        return self._handle_file_response(last_response_from_attack_target, feedback)

    async def _get_prompt_from_red_teaming_target(
        self,
        *,
        objective: str,
        prompt_target_conversation_id: str,
        red_teaming_chat_conversation_id: str,
        feedback: Optional[str] = None,
    ) -> str:
        """
        Send a prompt to the red teaming chat to generate a new prompt for the prompt target.

        Args:
            prompt_target_conversation_id (str): the conversation ID for the prompt target.
            red_teaming_chat_conversation_id (str): the conversation ID for the red teaming chat.
            feedback (str, optional): feedback from a previous iteration of the conversation.
                This can either be a score if the request completed, or a short prompt to rewrite
                the input if the request was blocked.
                The feedback is passed back to the red teaming chat to improve the next prompt.
                For text-to-image applications, for example, there is no immediate text output
                that can be passed back to the red teaming chat, so the scorer rationale is the
                only way to generate feedback.
        """
        prompt_text = self._get_prompt_for_red_teaming_chat(
            prompt_target_conversation_id=prompt_target_conversation_id, feedback=feedback
        )

        if len(self._memory.get_conversation(conversation_id=red_teaming_chat_conversation_id)) == 0:
            # Set system prompt for the first turn with red teaming chat
            system_prompt = SystemPromptWithObjective(
                path=self._red_team_target_system_prompt_path,
                objective=objective,
            )

            self._red_team_target.set_system_prompt(
                system_prompt=str(system_prompt),
                conversation_id=red_teaming_chat_conversation_id,
                orchestrator_identifier=self.get_identifier(),
                labels=self._global_memory_labels,
            )

        normalizer_request = self._create_normalizer_request(
            prompt_text=prompt_text, conversation_id=red_teaming_chat_conversation_id
        )

        response_text = (
            (
                await self._prompt_normalizer.send_prompt_async(
                    normalizer_request=normalizer_request,
                    target=self._red_team_target,
                    orchestrator_identifier=self.get_identifier(),
                    labels=self._global_memory_labels,
                )
            )
            .request_pieces[0]
            .converted_value
        )

        return response_text

    def _get_last_attack_target_response(self, prompt_target_conversation_id: str) -> PromptRequestPiece | None:
        target_messages = self._memory.get_conversation(conversation_id=prompt_target_conversation_id)
        assistant_responses = [m.request_pieces[0] for m in target_messages if m.request_pieces[0].role == "assistant"]
        return assistant_responses[-1] if len(assistant_responses) > 0 else None
