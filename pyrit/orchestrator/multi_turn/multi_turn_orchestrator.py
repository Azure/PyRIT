# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import asyncio
import logging

from abc import abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union
from colorama import Fore, Style

from pyrit.common.display_response import display_image_response
from pyrit.memory import CentralMemory
from pyrit.models import SeedPrompt
from pyrit.orchestrator import Orchestrator
from pyrit.prompt_normalizer import PromptNormalizer
from pyrit.prompt_target import PromptTarget, PromptChatTarget
from pyrit.prompt_converter import PromptConverter
from pyrit.score import Scorer

logger = logging.getLogger(__name__)


@dataclass
class MultiTurnAttackResult:
    """The result of a multi-turn attack."""

    def __init__(self, conversation_id: str, achieved_objective: bool, objective: str):
        self.conversation_id = conversation_id
        self.achieved_objective = achieved_objective
        self.objective = objective
        self._memory = CentralMemory.get_memory_instance()

    async def print_conversation_async(self):
        """Prints the conversation between the objective target and the adversarial chat, including the scores.

        Args:
            prompt_target_conversation_id (str): the conversation ID for the prompt target.
        """
        target_messages = self._memory._get_prompt_pieces_with_conversation_id(conversation_id=self.conversation_id)

        if not target_messages or len(target_messages) == 0:
            print("No conversation with the target")
            return

        if self.achieved_objective:
            print(
                f"{Style.BRIGHT}{Fore.RED}The multi-turn orchestrator has completed the conversation and achieved "
                f"the objective: {self.objective}"
            )
        else:
            print(
                f"{Style.BRIGHT}{Fore.RED}The multi-turn orchestrator has not achieved the objective: "
                f"{self.objective}"
            )

        for message in target_messages:
            if message.role == "user":
                print(f"{Style.BRIGHT}{Fore.BLUE}{message.role}:")
                if message.converted_value != message.original_value:
                    print(f"Original value: {message.original_value}")
                print(f"Converted value: {message.converted_value}")
            else:
                print(f"{Style.NORMAL}{Fore.YELLOW}{message.role}: {message.converted_value}")
                await display_image_response(message)

            scores = self._memory.get_scores_by_prompt_ids(prompt_request_response_ids=[str(message.id)])
            if scores and len(scores) > 0:
                for score in scores:
                    print(f"{Style.RESET_ALL}score: {score} : {score.score_rationale}")


class MultiTurnOrchestrator(Orchestrator):
    """
    The `MultiTurnOrchestrator` is an interface that coordinates attacks and conversations between
    a adversarial_chat target and an objective_target.

    Args:
        objective_target (PromptTarget): The target to send the created prompts to.
        adversarial_chat (PromptChatTarget): The endpoint that creates prompts that are sent to the
            objective_target.
        adversarial_chat_system_prompt_path (Path): The initial prompt to send to adversarial_chat.
        initial_adversarial_chat_prompt (str, Optional): The initial prompt to start the adversarial chat.
            Defaults to "How can I help you?".
        max_turns (int, Optional): The maximum number of turns for the conversation. Must be greater than or equal
            to 0. Defaults to 5.
        prompt_converters (Optional[list[PromptConverter]], Optional): The prompt converters to use to convert the
            prompts before sending them to the prompt target. Defaults to None.
        objective_scorer (Scorer): The scorer classifies the prompt target outputs as sufficient (True) or
            insufficient (False) to satisfy the objective that is specified in the attack_strategy.
        memory_labels (Optional[dict[str, str]], Optional): A free-form dictionary for tagging prompts with custom
            labels. These labels can be used to track all prompts sent as part of an operation, score prompts based
            on the operation ID (op_id), and tag each prompt with the relevant Responsible AI (RAI) harm category.
            Users can define any key-value pairs according to their needs. Defaults to None.
        verbose (bool, Optional): Whether to print debug information. Defaults to False.

    Raises:
        FileNotFoundError: If the file specified by adversarial_chat_system_prompt_path does not exist.
        ValueError: If max_turns is less than or equal to 0.
        ValueError: If the objective_scorer is not a true/false scorer.
    """

    def __init__(
        self,
        *,
        objective_target: PromptTarget,
        adversarial_chat: PromptChatTarget,
        adversarial_chat_system_prompt_path: Union[Path | str],
        initial_adversarial_chat_prompt: str = "How can I help you?",
        max_turns: int = 5,
        prompt_converters: Optional[list[PromptConverter]] = None,
        objective_scorer: Scorer,
        memory_labels: Optional[dict[str, str]] = None,
        verbose: bool = False,
    ) -> None:

        super().__init__(prompt_converters=prompt_converters, memory_labels=memory_labels, verbose=verbose)

        self._objective_target = objective_target
        self._achieved_objective = False

        self._adversarial_chat_system_seed_prompt = SeedPrompt.from_yaml_file(adversarial_chat_system_prompt_path)

        if "objective" not in self._adversarial_chat_system_seed_prompt.parameters:
            raise ValueError(f"Adversarial seed prompt must have an objective: '{adversarial_chat_system_prompt_path}'")

        self._prompt_normalizer = PromptNormalizer()
        self._adversarial_chat = adversarial_chat
        self._initial_adversarial_prompt = initial_adversarial_chat_prompt

        if max_turns <= 0:
            raise ValueError("The maximum number of turns must be greater than or equal to 0.")

        self._max_turns = max_turns

        if objective_scorer.scorer_type != "true_false":
            raise ValueError(
                f"The scorer must be a true/false scorer. The scorer type is {objective_scorer.scorer_type}."
            )
        self._objective_scorer = objective_scorer

    @abstractmethod
    async def run_attack_async(self, *, objective: str) -> MultiTurnAttackResult:
        """
        Applies the attack strategy until the conversation is complete or the maximum number of turns is reached.

        Args:
            objective (str): The specific goal the orchestrator aims to achieve through the conversation.

        Returns:
            MultiTurnAttackResult: Contains the outcome of the attack, including:
                - conversation_id (UUID): The ID associated with the final conversation state.
                - achieved_objective (bool): Indicates whether the orchestrator successfully met the objective.
                - objective (str): The intended goal of the attack.
        """

    async def run_attacks_async(self, *, objectives: list[str], batch_size=5) -> list[MultiTurnAttackResult]:
        """Applies the attack strategy for each objective in the list of objectives.

        Args:
            objectives: The list of objectives to apply the attack strategy.
            batch_size: The number of objectives to process in parallel. The default value is 5.

        Returns:
            list[MultiTurnAttackResult]: The list of MultiTurnAttackResults for each objective.
        """
        semaphore = asyncio.Semaphore(batch_size)

        async def limited_run_attack(objective):
            async with semaphore:
                return await self.run_attack_async(objective=objective)

        tasks = [limited_run_attack(objective) for objective in objectives]
        results = await asyncio.gather(*tasks)
        return results
