# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import asyncio
import logging
import uuid
from abc import abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Optional, Sequence, Union

from colorama import Fore, Style

from pyrit.common import combine_dict
from pyrit.common.display_response import display_image_response
from pyrit.memory import CentralMemory
from pyrit.models import PromptRequestPiece, PromptRequestResponse, Score, SeedPrompt
from pyrit.models.attack_configuration import AttackConfiguration
from pyrit.orchestrator import Orchestrator
from pyrit.prompt_converter import PromptConverter
from pyrit.prompt_normalizer import PromptNormalizer
from pyrit.prompt_target import PromptChatTarget, PromptTarget
from pyrit.score import Scorer

logger = logging.getLogger(__name__)


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
        target_messages = self._memory.get_conversation(conversation_id=self.conversation_id)

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
            for piece in message.request_pieces:
                if piece.role == "user":
                    print(f"{Style.BRIGHT}{Fore.BLUE}{piece.role}:")
                    if piece.converted_value != piece.original_value:
                        print(f"Original value: {piece.original_value}")
                    print(f"Converted value: {piece.converted_value}")
                else:
                    print(f"{Style.NORMAL}{Fore.YELLOW}{piece.role}: {piece.converted_value}")

                await display_image_response(piece)

                scores = self._memory.get_scores_by_prompt_ids(prompt_request_response_ids=[str(piece.id)])
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
        adversarial_chat_seed_prompt: Union[str | SeedPrompt] = "How can I help you?",
        max_turns: int = 5,
        prompt_converters: Optional[list[PromptConverter]] = None,
        objective_scorer: Scorer,
        verbose: bool = False,
    ) -> None:

        super().__init__(prompt_converters=prompt_converters, verbose=verbose)

        self._objective_target = objective_target
        self._achieved_objective = False

        self._adversarial_chat_system_seed_prompt = SeedPrompt.from_yaml_file(adversarial_chat_system_prompt_path)

        if "objective" not in self._adversarial_chat_system_seed_prompt.parameters:
            raise ValueError(f"Adversarial seed prompt must have an objective: '{adversarial_chat_system_prompt_path}'")

        self._prompt_normalizer = PromptNormalizer()
        self._adversarial_chat = adversarial_chat

        self._adversarial_chat_seed_prompt = self._get_adversarial_chat_seed_prompt(adversarial_chat_seed_prompt)

        if max_turns <= 0:
            raise ValueError("The maximum number of turns must be greater than or equal to 0.")

        self._max_turns = max_turns

        self._objective_scorer = objective_scorer

        self._prepended_conversation: list[PromptRequestResponse] = []
        self._last_prepended_user_message: str = ""
        self._last_prepended_assistant_message_scores: Sequence[Score] = []

    def _get_adversarial_chat_seed_prompt(self, seed_prompt):
        if isinstance(seed_prompt, str):
            return SeedPrompt(
                value=seed_prompt,
                data_type="text",
            )
        return seed_prompt

    @abstractmethod
    async def run_attack_async(
        self, *, attack_configuration: AttackConfiguration, memory_labels: Optional[dict[str, str]] = None
    ) -> MultiTurnAttackResult:
        """
        Applies the attack strategy until the conversation is complete or the maximum number of turns is reached.

        Args:
            attack_configuration: The attack configuration for this attack
            memory_labels (dict[str, str], Optional): A free-form dictionary of additional labels to apply to the
                prompts throughout the attack. Any labels passed in will be combined with self._global_memory_labels
                (from the GLOBAL_MEMORY_LABELS environment variable) into one dictionary. In the case of collisions,
                the passed-in labels take precedence. Defaults to None.

        Returns:
            MultiTurnAttackResult: Contains the outcome of the attack, including:
                - conversation_id (UUID): The ID associated with the final conversation state.
                - achieved_objective (bool): Indicates whether the orchestrator successfully met the objective.
                - objective (str): The intended goal of the attack.
        """

    async def run_attacks_async(
        self, *, objectives: list[str], memory_labels: Optional[dict[str, str]] = None, batch_size=5
    ) -> list[MultiTurnAttackResult]:
        """Applies the attack strategy for each objective in the list of objectives.

        Args:
            objectives (list[str]): The list of objectives to apply the attack strategy.
            memory_labels (dict[str, str], Optional): A free-form dictionary of additional labels to apply to the
                prompts throughout the attack. Any labels passed in will be combined with self._global_memory_labels
                (from the GLOBAL_MEMORY_LABELS environment variable) into one dictionary. In the case of collisions,
                the passed-in labels take precedence. Defaults to None.
            batch_size (int): The number of objectives to process in parallel. The default value is 5.

        Returns:
            list[MultiTurnAttackResult]: The list of MultiTurnAttackResults for each objective.
        """
        semaphore = asyncio.Semaphore(batch_size)

        async def limited_run_attack(objective):
            async with semaphore:
                updated_memory_labels = combine_dict(existing_dict=self._global_memory_labels, new_dict=memory_labels)
                attack_configuration = AttackConfiguration(
                    orchestrator_identifier=self.get_identifier(),
                    conversation_objective=objective,
                    labels=updated_memory_labels,
                    start_time=datetime.now()
                )
                self._memory.add_attack_configuration_to_memory(attack_configuration=attack_configuration)

                result = await self.run_attack_async(attack_configuration=attack_configuration, memory_labels=memory_labels)

                fields_to_update = {
                    "attack_result": {
                        "objective_achieved": result.achieved_objective
                    },
                    "end_time": datetime.now()
                }
                self._memory.update_attack_configuration(attack_id=attack_configuration.id, update_fields=fields_to_update)
                return result

        tasks = [limited_run_attack(objective) for objective in objectives]
        results = await asyncio.gather(*tasks)
        return results

    def set_prepended_conversation(self, *, prepended_conversation: list[PromptRequestResponse]):
        """Sets the prepended conversation to be sent to the objective target.
        This can be used to set the system prompt of the objective target, or send a series of
        user/assistant messages from which the orchestrator should start the conversation from.

        Args:
            prepended_conversation (str): The prepended conversation to send to the objective target.
        """
        self._prepended_conversation = prepended_conversation

    def _set_globals_based_on_role(self, last_message: PromptRequestPiece):
        """Sets the global variables of self._last_prepended_user_message and self._last_prepended_assistant_message
        based on the role of the last message in the prepended conversation.
        """
        # There is specific handling per orchestrator depending on the last message
        if last_message.role == "user":
            self._last_prepended_user_message = last_message.converted_value
        elif last_message.role == "assistant":
            # Get scores for the last assistant message based off of the original id
            self._last_prepended_assistant_message_scores = self._memory.get_scores_by_prompt_ids(
                prompt_request_response_ids=[str(last_message.original_prompt_id)]
            )

            # Do not set last user message if there are no scores for the last assistant message
            if not self._last_prepended_assistant_message_scores:
                return

            # Check assumption that there will be a user message preceding the assistant message
            if (
                len(self._prepended_conversation) > 1
                and self._prepended_conversation[-2].request_pieces[0].role == "user"
            ):
                self._last_prepended_user_message = self._prepended_conversation[-2].request_pieces[0].converted_value
            else:
                raise ValueError(
                    "There must be a user message preceding the assistant message in prepended conversations."
                )

    def _prepare_conversation(self, *, new_conversation_id: str, attack_configuration: AttackConfiguration) -> int:
        """Prepare the conversation by saving the prepended conversation to memory
        with the new conversation ID. This should only be called by inheriting classes.

        Args:
            new_conversation_id (str): The ID for the new conversation.

        Returns:
            turn_count (int): The turn number to start from, which counts
                the number of turns in the prepended conversation. E.g. with 2
                prepended conversations, the next turn would be turn 3. With no
                prepended conversations, the next turn is at 1. Value used by the
                calling orchestrators to set turn count.

        Raises:
            ValueError: If the number of turns in the prepended conversation equals or exceeds
                        the maximum number of turns.
            ValueError: If the objective target is not a PromptChatTarget, as PromptTargets do
                        not support setting system prompts.
        """
        logger.log(level=logging.INFO, msg=f"Preparing conversation with ID: {new_conversation_id}")

        turn_count = 1
        skip_iter = -1
        if self._prepended_conversation:
            last_message = self._prepended_conversation[-1].request_pieces[0]
            if last_message.role == "user":
                # Do not add last user message to memory as it will be added when sent
                # to the objective target by the orchestrator
                skip_iter = len(self._prepended_conversation) - 1

            for i, request in enumerate(self._prepended_conversation):
                add_to_memory = True

                for piece in request.request_pieces:
                    piece.conversation_id = new_conversation_id
                    piece.id = uuid.uuid4()
                    piece.orchestrator_identifier = self.get_identifier()

                    if piece.role == "system":
                        # Attempt to set system message if Objective Target is a PromptChatTarget
                        # otherwise throw exception
                        if isinstance(self._objective_target, PromptChatTarget):
                            self._objective_target.set_system_prompt(
                                system_prompt=piece.converted_value,
                                conversation_id=new_conversation_id,
                                orchestrator_identifier=piece.orchestrator_identifier,
                                attack_configuration=attack_configuration,
                                labels=piece.labels,
                            )

                            add_to_memory = False
                        else:
                            raise ValueError("Objective Target must be a PromptChatTarget to set system prompt.")
                    elif piece.role == "assistant":
                        # Number of complete turns should be the same as the number of assistant messages
                        turn_count += 1

                        if turn_count > self._max_turns:
                            raise ValueError(
                                f"The number of turns in the prepended conversation ({turn_count-1}) is equal to"
                                + f" or exceeds the maximum number of turns ({self._max_turns}), which means the"
                                + " conversation will not be able to continue. Please reduce the number of turns in"
                                + " the prepended conversation or increase the maximum number of turns and try again."
                            )

                if not add_to_memory or i == skip_iter:
                    continue

                self._memory.add_request_response_to_memory(request=request)

            self._set_globals_based_on_role(last_message=last_message)

        return turn_count
