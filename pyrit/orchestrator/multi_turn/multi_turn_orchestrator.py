# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from abc import abstractmethod
import asyncio
from dataclasses import dataclass
import logging
import os
from pathlib import Path
from typing import Optional
from colorama import Fore, Style

from pyrit.common.display_response import display_image_response
from pyrit.memory import MemoryInterface
from pyrit.orchestrator import Orchestrator
from pyrit.prompt_normalizer import PromptNormalizer
from pyrit.prompt_target import PromptTarget, PromptChatTarget
from pyrit.prompt_converter import PromptConverter
from pyrit.score import Scorer

logger = logging.getLogger(__name__)


@dataclass
class MultiTurnAttackResult:
    """The result of a multi-turn attack."""

    conversation_id: str
    achieved_objective: bool
    objective: str


class MultiTurnOrchestrator(Orchestrator):
    _memory: MemoryInterface

    def __init__(
        self,
        *,
        prompt_target: PromptTarget,
        red_team_target: PromptChatTarget,
        red_team_target_system_prompt_path: Path,
        initial_red_teaming_prompt: str = "How can I help you?",
        max_turns: int = 5,
        prompt_converters: Optional[list[PromptConverter]] = None,
        objective_scorer: Scorer,
        memory: Optional[MemoryInterface] = None,
        memory_labels: Optional[dict[str, str]] = None,
        verbose: bool = False,
    ) -> None:
        """Creates an orchestrator to manage conversations between a red teaming target and a prompt target.

        Args:
            prompt_target: The target to send the created prompts to.
            red_team_target: The endpoint that creates prompts that are sent to the prompt target.
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

        if not os.path.isfile(red_team_target_system_prompt_path):
            raise FileNotFoundError(f"The file '{red_team_target_system_prompt_path}' does not exist.")

        # TODO validate the yaml doesn't have any parameters besides conversation_objective (or it can be blank)

        self._red_team_target_system_prompt_path = red_team_target_system_prompt_path

        self._prompt_normalizer = PromptNormalizer(memory=self._memory)
        self._prompt_target._memory = self._memory
        self._red_team_target = red_team_target
        self._red_team_target._memory = self._memory
        self._initial_red_teaming_prompt = initial_red_teaming_prompt

        if max_turns <= 0:
            raise ValueError("The maximum number of turns must be greater than or equal to 0.")

        self._max_turns = max_turns

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
    async def run_attack_async(self, *, objective: str) -> MultiTurnAttackResult:
        """
        Applies the attack strategy until the conversation is complete or the maximum number of turns is reached.

        Args:
            max_turns: The maximum number of turns to apply the attack strategy.
                If the conversation is not complete after the maximum number of turns,
                the orchestrator stops and returns the last score.
                The default value is 5.

        Returns:
            MultiTurnAttackResult: The conversation ID for the final or successful multi-turn conversation and
                whether the orchestrator achieved the objective.
        """

    async def run_attacks_async(self, *, objectives: list[str], batch_size=5) -> list[MultiTurnAttackResult]:
        """Applies the attack strategy for each objective in the list of objectives.

        Args:
            objectives: The list of objectives to apply the attack strategy.

        Returns:
            list[MultiTurnAttackResult]: The list of conversation IDs for the final or successful multi-turn
                conversations and whether the orchestrator achieved the objective for each objective.
        """
        semaphore = asyncio.Semaphore(batch_size)

        async def limited_run_attack(objective):
            async with semaphore:
                return await self.run_attack_async(objective=objective)

        tasks = [limited_run_attack(objective) for objective in objectives]
        results = await asyncio.gather(*tasks)
        return results

    async def print_conversation_async(self, result: MultiTurnAttackResult):
        """Prints the conversation between the prompt target and the red teaming bot, including the scores.

        Args:
            prompt_target_conversation_id (str): the conversation ID for the prompt target.
        """
        target_messages = self._memory._get_prompt_pieces_with_conversation_id(conversation_id=result.conversation_id)

        if not target_messages or len(target_messages) == 0:
            print("No conversation with the target")
            return

        if result.achieved_objective:
            print(
                f"{Style.BRIGHT}{Fore.RED}The multi-turn orchestrator has completed the conversation and achieved "
                f"the objective: {result.objective}"
            )
        else:
            print(
                f"{Style.BRIGHT}{Fore.RED}The multi-turn orchestrator has not achieved the objective: "
                f"{result.objective}"
            )

        for message in target_messages:
            if message.role == "user":
                print(f"{Style.BRIGHT}{Fore.BLUE}{message.role}:")
                if message.converted_value != message.original_value:
                    print(f"Original value: {message.original_value}")
                print(f"Converted value: {message.converted_value}")
            else:
                print(f"{Style.NORMAL}{Fore.YELLOW}{message.role}: {message.converted_value}")
                await display_image_response(message, self._memory)

            scores = self._memory.get_scores_by_prompt_ids(prompt_request_response_ids=[str(message.id)])
            if scores and len(scores) > 0:
                for score in scores:
                    print(f"{Style.RESET_ALL}score: {score} : {score.score_rationale}")
