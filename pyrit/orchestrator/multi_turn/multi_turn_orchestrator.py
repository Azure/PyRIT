# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import asyncio
import logging
import uuid
from abc import abstractmethod
from pathlib import Path
from typing import Optional, Sequence, Union

from pyrit.models import PromptRequestPiece, PromptRequestResponse, Score, SeedPrompt
from pyrit.orchestrator import Orchestrator, OrchestratorResult
from pyrit.prompt_converter import PromptConverter
from pyrit.prompt_normalizer import PromptNormalizer
from pyrit.prompt_target import PromptChatTarget, PromptTarget
from pyrit.score import Scorer

logger = logging.getLogger(__name__)


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
        batch_size (int, Optional): The number of objectives to process in parallel. The default value is 1.
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
        batch_size: int = 1,
        verbose: bool = False,
    ) -> None:

        super().__init__(prompt_converters=prompt_converters, verbose=verbose)

        self._objective_target = objective_target

        self._adversarial_chat_system_seed_prompt = SeedPrompt.from_yaml_file(adversarial_chat_system_prompt_path)

        if "objective" not in self._adversarial_chat_system_seed_prompt.parameters:
            raise ValueError(f"Adversarial seed prompt must have an objective: '{adversarial_chat_system_prompt_path}'")

        self._prompt_normalizer = PromptNormalizer()
        self._adversarial_chat = adversarial_chat

        self._adversarial_chat_seed_prompt = self._get_adversarial_chat_seed_prompt(adversarial_chat_seed_prompt)

        self._batch_size = batch_size

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
        self, *, objective: str, memory_labels: Optional[dict[str, str]] = None
    ) -> OrchestratorResult:
        """
        Applies the attack strategy until the conversation is complete or the maximum number of turns is reached.

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
        """

    async def run_attacks_async(
        self, *, objectives: list[str], memory_labels: Optional[dict[str, str]] = None
    ) -> list[OrchestratorResult]:
        """Applies the attack strategy for each objective in the list of objectives.

        Args:
            objectives (list[str]): The list of objectives to apply the attack strategy.
            memory_labels (dict[str, str], Optional): A free-form dictionary of additional labels to apply to the
                prompts throughout the attack. Any labels passed in will be combined with self._global_memory_labels
                (from the GLOBAL_MEMORY_LABELS environment variable) into one dictionary. In the case of collisions,
                the passed-in labels take precedence. Defaults to None.

        Returns:
            list[OrchestratorResult]: The list of OrchestratorResults for each objective, each containing:
                - conversation_id (str): The ID associated with the final conversation state.
                - objective (str): The intended goal of the attack.
                - status (OrchestratorResultStatus): The status of the attack ("success", "failure", "pruned", etc.)
                - score (Score): The score evaluating the attack outcome.
                - confidence (float): The confidence level of the result.
        """
        semaphore = asyncio.Semaphore(self._batch_size)

        async def limited_run_attack(objective):
            async with semaphore:
                return await self.run_attack_async(objective=objective, memory_labels=memory_labels)

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
                self._last_prepended_user_message = self._prepended_conversation[-2].get_value()
            else:
                raise ValueError(
                    "There must be a user message preceding the assistant message in prepended conversations."
                )

    def _prepare_conversation(self, *, new_conversation_id: str) -> int:
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
