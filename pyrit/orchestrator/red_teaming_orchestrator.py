# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging
from typing import Optional, Union
from uuid import uuid4
from PIL import Image

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

        self._prompt_normalizer = PromptNormalizer(memory=self._memory)
        self._prompt_target._memory = self._memory
        self._prompt_target_conversation_id = str(uuid4())
        self._red_teaming_chat_conversation_id = str(uuid4())
        self._red_teaming_chat = red_teaming_chat
        self._red_teaming_chat._memory = self._memory
        self._attack_strategy = str(attack_strategy)
        self._initial_red_teaming_prompt = initial_red_teaming_prompt
        self._use_score_as_feedback = use_score_as_feedback
        if scorer.scorer_type != "true_false":
            raise ValueError(f"The scorer must be a true/false scorer. The scorer type is {scorer.scorer_type}.")
        self._scorer = scorer
        
    async def check_conversation_complete_async(self) -> bool:
        """
        Returns True if the conversation is complete, False otherwise.
        This function uses the scorer to classify the last response.
        """
        prompt_request_responses = self._memory.get_conversation(
            conversation_id=self._prompt_target_conversation_id
        )
        if not prompt_request_responses:
            # If there are no messages, then the conversation is not complete.
            return False
        if prompt_request_responses[-1].request_pieces[0].role in ["user", "system"]:
            # If the last message is a system or red teaming chat message,
            # then the conversation is not yet complete.
            return False

        last_message_content = prompt_request_responses[-1].request_pieces[0].converted_value
        last_message_type = prompt_request_responses[-1].request_pieces[0].converted_value_data_type
        if last_message_type == "text":
            score = (await self._scorer.score_text_async(text=last_message_content))[0]
        elif last_message_type == "image_path":
            score = (await self._scorer.score_image_async(image_path=last_message_content))[0]
        else:
            raise ValueError(f"Unsupported message type: {last_message_type}")
        
        if score.score_type != "true_false":
            raise ValueError(f"The scorer must return a true_false score. The score type is {score.score_type}.")
        return score.get_value()

    async def apply_attack_strategy_until_completion_async(self, *, max_turns: int = 5):
        """
        Applies the attack strategy until the conversation is complete or the maximum number of turns is reached.

        Args:
            max_turns: The maximum number of turns to apply the attack strategy.
                If the conversation is not complete after the maximum number of turns,
                the orchestrator stops and returns the last score.
                The default value is 5.
        """
        turn = 1
        success = False
        score: Score | None = None
        blocked = False
        while turn <= max_turns:
            logger.info(f"Applying the attack strategy for turn {turn}.")

            send_prompt_kwargs = {}
            if self._use_score_as_feedback and score:
                send_prompt_kwargs["score_explanation"] = score.score_explanation
            if blocked:
                send_prompt_kwargs["blocked"] = "Request to target failed: blocked. Please rewrite your prompt to avoid getting blocked next time."
                blocked = False

            response = self.send_prompt(**send_prompt_kwargs)
            self._display_response(response)

            # Check if there was an error. No error or blocked error is acceptable
            # since the red teaming chat can still continue the conversation.
            if response.response_error not in ["none", "blocked"]:
                raise RuntimeError(f"Response error: {response.response_error}")

            if response.response_error == "none":
                score = await self.check_conversation_complete_async()
                if bool(score.score_value):
                    success = True
                    logger.info(
                        "The red teaming orchestrator has completed the conversation and achieved the objective.",
                    )
                    break
            else:
                score = None
            turn += 1

        if not success:
            logger.info(
                "The red teaming orchestrator has not achieved the objective after the maximum "
                f"number of turns ({max_turns}).",
            )

        return score
    
    def _display_response(self, response_piece: PromptRequestPiece) -> None:
        # If running in notebook environment, display the image.
        if response_piece.response_error == "none" and response_piece.converted_value_data_type == "image_path" and is_in_ipython_session():
            with open(response_piece.converted_value, "rb") as f:
                img = Image.open(f)
                # Jupyter built-in display function only works in notebooks.
                display(img)
        if response_piece.response_error == "blocked":
            print("---\nContent blocked, cannot show a response.\n---")

    def send_prompt(self, *, prompt: Optional[str] = None, score_explanation: Optional[str] = None) -> PromptRequestPiece:
        """
        Either sends a user-provided prompt or generates a prompt to send to the prompt target.

        Args:
            prompt: The prompt to send to the target.
                If no prompt is specified the orchestrator contacts the red teaming target
                to generate a prompt and forwards it to the prompt target.
                This can only be specified for the first iteration at this point.
            score_explanation: The explanation of the score to send to the red teaming chat.
                This is feedback from a previous iteration of send_prompt that is determined
                through a scorer and passed back to the red teaming chat to improve the next prompt.
                For text-to-text applications, this may not be needed.
                For text-to-image applications, for example, there is no immediate text output
                that can be passed back to the red teaming chat, so the scorer explanation is the
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
            logger.info(
                "Generating a prompt for the prompt target using the red teaming LLM."
            )
            prompt = self._get_prompt_from_red_teaming_target(score_explanation=score_explanation)

        target_prompt_obj = NormalizerRequestPiece(
            prompt_converters=self._prompt_converters,
            prompt_text=prompt,
            prompt_data_type="text",
        )

        response_piece = (
            self._prompt_normalizer.send_prompt(
                normalizer_request=NormalizerRequest([target_prompt_obj]),
                target=self._prompt_target,
                conversation_id=self._prompt_target_conversation_id,
                labels=self._global_memory_labels,
                orchestrator_identifier=self.get_identifier(),
            )
            .request_pieces[0]
        )

        return response_piece

    def _get_prompt_from_red_teaming_target(self, *, score_explanation: Optional[str] = None) -> str:
        # If we have previously exchanged messages with the attack target
        # we can use the last message from the attack target as the new
        # prompt for the red teaming chat.
        # If there is no response from the attack target (i.e., this is the first turn),
        # we use the initial red teaming prompt.
        last_response_from_attack_target = self._get_last_attack_target_response()
        if not last_response_from_attack_target:
            logger.info(f"Using the specified initial red teaming prompt: {self._initial_red_teaming_prompt}")
            prompt_text = self._initial_red_teaming_prompt
        else:
            if last_response_from_attack_target.response_error != "none":
                prompt_text = f"Request to target failed: {last_response_from_attack_target.response_error}"
            else:
                # A successful response from the attack target exists.
                # If the attack target responds with text we can use that as the new prompt.
                # Otherwise, we need to rely on the scorer's (textual) explanation.
                if last_response_from_attack_target.converted_value_data_type == "text":
                    prompt_text = last_response_from_attack_target.converted_value
                elif self._use_score_as_feedback and score_explanation:
                    prompt_text = score_explanation
                else:
                    base_error_message = (
                        "The attack target does not respond with text output, "
                        "so the scorer explanation is the only textual feedback "
                        "that can be passed to the red teaming chat. "
                    )
                    if not self._use_score_as_feedback:
                        raise ValueError(
                            f"{base_error_message}"
                            "However, the use_score_as_feedback flag is set to False."
                        )
                    raise ValueError(
                        f"{base_error_message}"
                        "However, no score_explanation was provided."
                    )

        if self._is_first_turn_with_red_teaming_chat():
            self._red_teaming_chat.set_system_prompt(
                system_prompt=self._attack_strategy,
                conversation_id=self._red_teaming_chat_conversation_id,
                orchestrator_identifier=self.get_identifier(),
                labels=self._global_memory_labels,
            )

        response_text = (
            self._red_teaming_chat.send_chat_prompt(
                prompt=prompt_text,
                conversation_id=self._red_teaming_chat_conversation_id,
                orchestrator_identifier=self.get_identifier(),
                labels=self._global_memory_labels,
            )
            .request_pieces[0]
            .converted_value
        )

        return response_text

    def _get_last_attack_target_response(self) -> PromptRequestPiece | None:
        target_messages = self._memory.get_conversation(
            conversation_id=self._prompt_target_conversation_id
        )
        assistant_responses = [
            m.request_pieces[0] for m in target_messages
            if m.request_pieces[0].role == "assistant"
        ]
        return assistant_responses[-1] if len(assistant_responses) > 0 else None

    def _is_first_turn_with_red_teaming_chat(self):
        red_teaming_chat_messages = self._memory.get_chat_messages_with_conversation_id(
            conversation_id=self._red_teaming_chat_conversation_id
        )
        return len(red_teaming_chat_messages) == 0
