# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging
from typing import Optional, Union
from pyrit.interfaces import SupportTextClassification

from pyrit.memory import MemoryInterface
from pyrit.models import AttackStrategy, ChatMessage
from pyrit.orchestrator.base_red_teaming_orchestrator import BaseRedTeamingOrchestrator
from pyrit.prompt_target import PromptTarget
from pyrit.prompt_converter import PromptConverter


logger = logging.getLogger(__name__)
    

class ScoringRedTeamingOrchestrator(BaseRedTeamingOrchestrator):
    def __init__(
        self,
        *,
        attack_strategy: Union[str, AttackStrategy],
        prompt_target: PromptTarget,
        red_teaming_target: PromptTarget,
        initial_red_teaming_prompt: str,
        scorer: SupportTextClassification,
        prompt_converter: Optional[PromptConverter] = None,
        memory: Optional[MemoryInterface] = None,
        memory_labels: list[str] = ["red-teaming-orchestrator"],
        verbose: bool = False,
    ) -> None:
        """Creates an orchestrator to manage conversations between a red teaming target and a prompt target.

        Args:
            attack_strategy: The attack strategy to follow by the bot. This can be used to guide the bot to achieve
                the conversation objective in a more direct and structured way. It is a string that can be written in
                a single sentence or paragraph. If not provided, the bot will use red_team_chatbot_with_objective.
                Should be of type string or AttackStrategy (which has a __str__ method).
            prompt_target: The target to send the prompts to.
            red_teaming_target: The endpoint that creates prompts that are sent to the prompt target.
            initial_red_teaming_prompt: The initial prompt to send to the red teaming target.
                The attack_strategy only provides the strategy, but not the starting point of the conversation.
                The initial_red_teaming_prompt is used to start the conversation with the red teaming target.
            scorer: The scorer classifies the prompt target outputs as sufficient (True) or insufficient (False)
                to satisfy the objective that is specified in the attack_strategy.
            prompt_converter: The prompt converter to use to convert the prompts before sending them to the prompt
                target. The converter is not applied in messages to the red teaming target.
            memory: The memory to use to store the chat messages. If not provided, a FileMemory will be used.
            memory_labels: The labels to use for the memory. This is useful to identify the bot messages in the memory.
            verbose: Whether to print debug information.
        """
        super().__init__(
            attack_strategy=attack_strategy,
            prompt_target=prompt_target,
            red_teaming_target=red_teaming_target,
            initial_red_teaming_prompt=initial_red_teaming_prompt,
            prompt_converter=prompt_converter,
            memory=memory,
            memory_labels=memory_labels,
            verbose=verbose,
        )
        self._scorer = scorer

    def is_conversation_complete(self, messages: list[ChatMessage]) -> bool:
        """
        Returns True if the conversation is complete, False otherwise.
        This function uses the scorer to classify the last response.
        """
        if not messages:
            # If there are no messages, then the conversation is not complete.
            return False
        if messages[-1].role == "system":
            # If the last message is a system message, then the conversation is not yet complete.
            return False
        score = self._scorer.score_text(messages[-1].content)
        if score.score_type != "bool":
            raise ValueError(f"The scorer must return a boolean score. The score type is {score.score_type}.")
        return score.score_value
