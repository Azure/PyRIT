# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from abc import abstractmethod
import logging
from typing import Optional
from uuid import UUID
from colorama import Fore, Style

from pyrit.common.display_response import display_image_response
from pyrit.memory import MemoryInterface
from pyrit.orchestrator import Orchestrator
from pyrit.prompt_normalizer import PromptNormalizer
from pyrit.prompt_target import PromptTarget, PromptChatTarget
from pyrit.prompt_converter import PromptConverter
from pyrit.score import Scorer

logger = logging.getLogger(__name__)


class MultiTurnOrchestrator(Orchestrator):
    _memory: MemoryInterface

    def __init__(
        self,
        *,
        prompt_target: PromptTarget,
        red_teaming_chat: PromptChatTarget,
        initial_red_teaming_prompt: Optional[str] = "How can I help you?",
        prompt_converters: Optional[list[PromptConverter]] = None,
        objective_scorer: Scorer,
        memory: Optional[MemoryInterface] = None,
        memory_labels: Optional[dict[str, str]] = None,
        verbose: bool = False,
    ) -> None:
        """Creates an orchestrator to manage conversations between a red teaming target and a prompt target.

        Args:
            prompt_target: The target to send the created prompts to.
            red_teaming_chat: The endpoint that creates prompts that are sent to the prompt target.
            initial_red_teaming_prompt: The initial prompt to send to the red teaming target.
                The attack_strategy only provides the strategy, but not the starting point of the conversation.
                The initial_red_teaming_prompt is used to start the conversation with the red teaming target.
                The default is a text prompt with the content "Begin Conversation".
            prompt_converters: The prompt converters to use to convert the prompts before sending them to the prompt
                target. The converters are not applied on messages to the red teaming target.
            objective_scorer: The scorer classifies the prompt target outputs as sufficient (True) or insufficient
                (False) to satisfy the objective that is specified in the attack_strategy.
            memory: The memory to use to store the chat messages. If not provided, a DuckDBMemory will be used.
            memory_labels (dict[str, str], optional): A free-form dictionary for tagging prompts with custom labels.
                These labels can be used to track all prompts sent as part of an operation, score prompts based on
                the operation ID (op_id), and tag each prompt with the relevant Responsible AI (RAI) harm category.
                Users can define any key-value pairs according to their needs. Defaults to None.
                verbose: Whether to print debug information.
        """

        super().__init__(
            prompt_converters=prompt_converters, memory=memory, memory_labels=memory_labels, verbose=verbose
        )

        self._prompt_target = prompt_target
        self._achieved_objective = False

        self._prompt_normalizer = PromptNormalizer(memory=self._memory)
        self._prompt_target._memory = self._memory
        self._red_teaming_chat = red_teaming_chat
        self._red_teaming_chat._memory = self._memory
        self._initial_red_teaming_prompt = initial_red_teaming_prompt
        if not self._initial_red_teaming_prompt:
            raise ValueError("The initial red teaming prompt cannot be empty.")
        if objective_scorer.scorer_type != "true_false":
            raise ValueError(
                f"The scorer must be a true/false scorer. The scorer type is {objective_scorer.scorer_type}."
            )
        self._objective_scorer = objective_scorer

        # Set the scorer and scorer._prompt_target memory to match the orchestrator's memory.
        if self._objective_scorer:
            self._objective_scorer._memory = self._memory
            if hasattr(self._objective_scorer, "_prompt_target"):
                self._objective_scorer._prompt_target._memory = self._memory

    @abstractmethod
    async def run_attack_async(
        self,
        *,
        max_turns: int = 5,
    ) -> UUID:
        """
        Applies the attack strategy until the conversation is complete or the maximum number of turns is reached.

        Args:
            max_turns: The maximum number of turns to apply the attack strategy.
                If the conversation is not complete after the maximum number of turns,
                the orchestrator stops and returns the last score.
                The default value is 5.

        Returns:
            conversation_id: The conversation ID for the final or successful multi-turn conversation.
        """

    async def print_conversation(self, prompt_target_conversation_id: str):
        """Prints the conversation between the prompt target and the red teaming bot, including the scores.

        Args:
            prompt_target_conversation_id (str): the conversation ID for the prompt target.
        """
        target_messages = self._memory._get_prompt_pieces_with_conversation_id(
            conversation_id=prompt_target_conversation_id
        )

        if not target_messages or len(target_messages) == 0:
            print("No conversation with the target")
            return

        if self._achieved_objective:
            print("The red teaming orchestrator has completed the conversation and achieved the objective.")
        else:
            print("The red teaming orchestrator has not achieved the objective.")

        for message in target_messages:
            if message.role == "user:":
                print(f"{Style.BRIGHT}{Fore.BLUE}{message.role}")
                if message.converted_value != message.original_value:
                    print(f"Original value: {message.original_value}")
                print(f"Converted value: {message.converted_value}")
            else:
                print(f"{Style.NORMAL}{Fore.YELLOW}{message.role}: {message.converted_value}")
                await display_image_response(message, self._memory)

            scores = self._memory.get_scores_by_prompt_ids(prompt_request_response_ids=[str(message.id)])
            if scores and len(scores) > 0:
                score = scores[0]
                print(f"{Style.RESET_ALL}score: {score} : {score.score_rationale}")
