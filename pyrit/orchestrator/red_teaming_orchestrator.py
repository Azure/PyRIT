# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging
from typing import Optional, Union, Dict, Any
from uuid import uuid4
from PIL import Image
from colorama import Fore, Style

from pyrit.common.notebook_utils import is_in_ipython_session
from pyrit.memory import MemoryInterface
from pyrit.models import AttackStrategy, PromptRequestPiece
from pyrit.orchestrator import Orchestrator
from pyrit.prompt_normalizer import NormalizerRequestPiece, PromptNormalizer, NormalizerRequest
from pyrit.prompt_target import PromptTarget, PromptChatTarget
from pyrit.prompt_converter import PromptConverter
from pyrit.score import Scorer, Score

logger = logging.getLogger(__name__)


class CompletionState:
    def __init__(self, is_complete: bool):
        self.is_complete = is_complete


class RedTeamingOrchestrator(Orchestrator):
    _memory: MemoryInterface

    def __init__(
        self,
        *,
        attack_strategy: Union[str, AttackStrategy],
        prompt_target: PromptTarget,
        red_teaming_chat: PromptChatTarget,
        initial_red_teaming_prompt: Optional[str] = "Begin conversation",
        prompt_converters: Optional[list[PromptConverter]] = None,
        scorer: Scorer,
        use_score_as_feedback: bool = False,
        memory: Optional[MemoryInterface] = None,
        memory_labels: Optional[dict[str, str]] = None,
        verbose: bool = False,
    ) -> None:
        """Creates an orchestrator to manage conversations between a red teaming target and a prompt target.

        Args:
            attack_strategy: The attack strategy for the red teaming bot to follow.
                It is used as the metaprompt in the conversation with the red teaming bot.
                This can be used to guide the bot to achieve the conversation objective in a more direct and
                structured way.
                Should be of type string or AttackStrategy (which has a __str__ method).
            prompt_target: The target to send the prompts to.
            red_teaming_chat: The endpoint that creates prompts that are sent to the prompt target.
            initial_red_teaming_prompt: The initial prompt to send to the red teaming target.
                The attack_strategy only provides the strategy, but not the starting point of the conversation.
                The initial_red_teaming_prompt is used to start the conversation with the red teaming target.
                The default is a text prompt with the content "Begin Conversation".
            prompt_converters: The prompt converters to use to convert the prompts before sending them to the prompt
                target. The converters are not applied on messages to the red teaming target.
                scorer: The scorer classifies the prompt target outputs as sufficient (True) or insufficient (False)
                to satisfy the objective that is specified in the attack_strategy.
            use_score_as_feedback: Whether to use the score as feedback to the red teaming chat.
            memory: The memory to use to store the chat messages. If not provided, a DuckDBMemory will be used.
            memory_labels: The labels to use for the memory. This is useful to identify the messages in the memory.
            verbose: Whether to print debug information.
        """

        super().__init__(
            prompt_converters=prompt_converters, memory=memory, memory_labels=memory_labels, verbose=verbose
        )

        self._prompt_target = prompt_target
        self._achieved_objective = False

        self._prompt_normalizer = PromptNormalizer(memory=self._memory)
        self._prompt_target._memory = self._memory
        self._prompt_target_conversation_id = str(uuid4())
        self._red_teaming_chat_conversation_id = str(uuid4())
        self._red_teaming_chat = red_teaming_chat
        self._red_teaming_chat._memory = self._memory
        self._attack_strategy = str(attack_strategy)
        self._initial_red_teaming_prompt = initial_red_teaming_prompt
        if not self._initial_red_teaming_prompt:
            raise ValueError("The initial red teaming prompt cannot be empty.")
        self._use_score_as_feedback = use_score_as_feedback
        if scorer.scorer_type != "true_false":
            raise ValueError(f"The scorer must be a true/false scorer. The scorer type is {scorer.scorer_type}.")
        self._scorer = scorer

    async def check_conversation_complete_async(self) -> Union[Score, None]:
        """
        Returns the scoring result of the conversation.
        This function uses the scorer to classify the last response.
        """
        prompt_request_responses = self._memory.get_conversation(conversation_id=self._prompt_target_conversation_id)
        if not prompt_request_responses:
            # If there are no messages, then the conversation is not complete.
            return None
        if prompt_request_responses[-1].request_pieces[0].role in ["user", "system"]:
            # If the last message is a system or red teaming chat message,
            # then the conversation is not yet complete.
            return None

        score = (await self._scorer.score_async(request_response=prompt_request_responses[-1].request_pieces[0]))[0]

        if score.score_type != "true_false":
            raise ValueError(f"The scorer must return a true_false score. The score type is {score.score_type}.")
        return score

    async def apply_attack_strategy_until_completion_async(self, *, max_turns: int = 5) -> Score:
        """
        Applies the attack strategy until the conversation is complete or the maximum number of turns is reached.

        Args:
            max_turns: The maximum number of turns to apply the attack strategy.
                If the conversation is not complete after the maximum number of turns,
                the orchestrator stops and returns the last score.
                The default value is 5.
        """
        turn = 1
        self._achieved_objective = False
        score: Score | None = None
        while turn <= max_turns:
            logger.info(f"Applying the attack strategy for turn {turn}.")

            send_prompt_kwargs: Dict[str, Any] = {}
            if self._use_score_as_feedback and score:
                send_prompt_kwargs["feedback"] = score.score_rationale

            response = await self.send_prompt_async(**send_prompt_kwargs)

            if response.response_error == "none":
                score = await self.check_conversation_complete_async()
                if bool(score.get_value()):
                    self._achieved_objective = True
                    logger.info(
                        "The red teaming orchestrator has completed the conversation and achieved the objective.",
                    )
                    break
            elif response.response_error == "blocked":
                score = None
            else:
                raise RuntimeError(f"Response error: {response.response_error}")

            turn += 1

        if not self._achieved_objective:
            logger.info(
                "The red teaming orchestrator has not achieved the objective after the maximum "
                f"number of turns ({max_turns}).",
            )

        return score

    def _display_response(self, response_piece: PromptRequestPiece) -> None:
        # If running in notebook environment, display the image.
        if (
            response_piece.response_error == "none"
            and response_piece.converted_value_data_type == "image_path"
            and is_in_ipython_session()
        ):
            with open(response_piece.converted_value, "rb") as f:
                img = Image.open(f)
                # Jupyter built-in display function only works in notebooks.
                display(img)  # type: ignore # noqa: F821
        if response_piece.response_error == "blocked":
            logger.info("---\nContent blocked, cannot show a response.\n---")

    async def send_prompt_async(
        self, *, prompt: Optional[str] = None, feedback: Optional[str] = None, blocked: bool = False
    ) -> PromptRequestPiece:
        """
        Either sends a user-provided prompt or generates a prompt to send to the prompt target.

        Args:
            prompt: The prompt to send to the target.
                If no prompt is specified the orchestrator contacts the red teaming target
                to generate a prompt and forwards it to the prompt target.
                This can only be specified for the first iteration at this point.
            feedback: feedback from a previous iteration of send_prompt_async.
                This can either be a score if the request completed, or a short prompt to rewrite
                the input if the request was blocked.
                The feedback is passed back to the red teaming chat to improve the next prompt.
                For text-to-image applications, for example, there is no immediate text output
                that can be passed back to the red teaming chat, so the scorer rationale is the
                only way to generate feedback.
        """
        target_messages = self._memory.get_chat_messages_with_conversation_id(
            conversation_id=self._prompt_target_conversation_id
        )
        if prompt:
            if target_messages:
                raise ValueError("The prompt argument can only be provided on the first iteration.")
            logger.info("Custom initial prompt provided.")
        else:
            # The prompt for the red teaming LLM needs to include the latest message from the prompt target.
            # A special case is the very first message, which means there are no prior messages.
            logger.info("Generating a prompt for the prompt target using the red teaming LLM.")
            prompt = await self._get_prompt_from_red_teaming_target(feedback=feedback)

        target_prompt_obj = NormalizerRequestPiece(
            request_converters=self._prompt_converters,
            prompt_value=prompt,
            prompt_data_type="text",
        )

        response_piece = (
            await self._prompt_normalizer.send_prompt_async(
                normalizer_request=NormalizerRequest([target_prompt_obj]),
                target=self._prompt_target,
                conversation_id=self._prompt_target_conversation_id,
                labels=self._global_memory_labels,
                orchestrator_identifier=self.get_identifier(),
            )
        ).request_pieces[0]

        return response_piece

    def print_conversation(self):
        """Prints the conversation between the prompt target and the red teaming bot."""
        target_messages = self._memory._get_prompt_pieces_with_conversation_id(
            conversation_id=self._prompt_target_conversation_id
        )

        if not target_messages or len(target_messages) == 0:
            print("No conversation with the target")
            return

        if self._achieved_objective:
            print("The red teaming orchestrator has completed the conversation and achieved the objective.")
        else:
            print("The red teaming orchestrator has not achieved the objective.")

        for message in target_messages:
            if message.role == "user":
                print(f"{Style.BRIGHT}{Fore.BLUE}{message.role}: {message.converted_value}")
            else:
                print(f"{Style.NORMAL}{Fore.YELLOW}{message.role}: {message.converted_value}")
                self._display_response(message)

            scores = self._memory.get_scores_by_prompt_ids(prompt_request_response_ids=[message.id])
            if scores and len(scores) > 0:
                score = scores[0]
                print(f"{Style.RESET_ALL}score: {score} : {score.score_rationale}")

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

    def _get_prompt_for_red_teaming_chat(self, *, feedback: str | None) -> str:
        # If we have previously exchanged messages with the attack target
        # we can use the last message from the attack target as the new
        # prompt for the red teaming chat.
        # If there is no response from the attack target (i.e., this is the first turn),
        # we use the initial red teaming prompt.
        last_response_from_attack_target = self._get_last_attack_target_response()
        if not last_response_from_attack_target:
            logger.info(f"Using the specified initial red teaming prompt: {self._initial_red_teaming_prompt}")
            return self._initial_red_teaming_prompt

        if last_response_from_attack_target.converted_value_data_type in ["text", "error"]:
            return self._handle_text_response(last_response_from_attack_target, feedback)

        return self._handle_file_response(last_response_from_attack_target, feedback)

    async def _get_prompt_from_red_teaming_target(self, *, feedback: Optional[str] = None) -> str:
        prompt_text = self._get_prompt_for_red_teaming_chat(feedback=feedback)

        if self._is_first_turn_with_red_teaming_chat():
            self._red_teaming_chat.set_system_prompt(
                system_prompt=self._attack_strategy,
                conversation_id=self._red_teaming_chat_conversation_id,
                orchestrator_identifier=self.get_identifier(),
                labels=self._global_memory_labels,
            )

        response_text = (
            (
                await self._prompt_normalizer.send_prompt_async(
                    normalizer_request=self._create_normalizer_request(prompt_text=prompt_text),
                    target=self._red_teaming_chat,
                    conversation_id=self._red_teaming_chat_conversation_id,
                    orchestrator_identifier=self.get_identifier(),
                    labels=self._global_memory_labels,
                )
            )
            .request_pieces[0]
            .converted_value
        )

        return response_text

    def _get_last_attack_target_response(self) -> PromptRequestPiece | None:
        target_messages = self._memory.get_conversation(conversation_id=self._prompt_target_conversation_id)
        assistant_responses = [m.request_pieces[0] for m in target_messages if m.request_pieces[0].role == "assistant"]
        return assistant_responses[-1] if len(assistant_responses) > 0 else None

    def _is_first_turn_with_red_teaming_chat(self):
        red_teaming_chat_messages = self._memory.get_chat_messages_with_conversation_id(
            conversation_id=self._red_teaming_chat_conversation_id
        )
        return len(red_teaming_chat_messages) == 0
