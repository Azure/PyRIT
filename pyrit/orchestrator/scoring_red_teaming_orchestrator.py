# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging
from typing import Optional, Union
from pyrit.interfaces import ChatSupport, SupportTextClassification

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
        red_teaming_chat: ChatSupport,
        initial_red_teaming_prompt: str = "Begin Conversation",
        scorer: SupportTextClassification,
        prompt_converters: Optional[list[PromptConverter]] = None,
        memory: Optional[MemoryInterface] = None,
        memory_labels: list[str] = ["red-teaming-orchestrator"],
        verbose: bool = False,
    ) -> None:
        """Creates an orchestrator to manage conversations between a red teaming bot and a prompt target.

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
            scorer: The scorer classifies the prompt target outputs as sufficient (True) or insufficient (False)
                to satisfy the objective that is specified in the attack_strategy.
            prompt_converters: The prompt converters to use to convert the prompts before sending them to the prompt
                target. The converters are not applied on messages to the red teaming target.
            memory: The memory to use to store the chat messages. If not provided, a FileMemory will be used.
            memory_labels: The labels to use for the memory. This is useful to identify the bot messages in the memory.
            verbose: Whether to print debug information.
        """
        super().__init__(
            attack_strategy=attack_strategy,
            prompt_target=prompt_target,
            red_teaming_chat=red_teaming_chat,
            initial_red_teaming_prompt=initial_red_teaming_prompt,
            prompt_converters=prompt_converters,
            memory=memory,
            memory_labels=memory_labels,
            verbose=verbose,
        )
        self._scorer = scorer

    def is_conversation_complete(self, messages: list[ChatMessage], *, red_teaming_chat_role) -> bool:
        """
        Returns True if the conversation is complete, False otherwise.
        This function uses the scorer to classify the last response.
        """
        if not messages:
            # If there are no messages, then the conversation is not complete.
            return False
        if messages[-1].role == red_teaming_chat_role or messages[-1].role == "system":
            # If the last message is a system or red teaming chat bot message,
            # then the conversation is not yet complete.
            return False
        score = self._scorer.score_text(text=messages[-1].content)
        if score.score_type != "bool":
            raise ValueError(f"The scorer must return a boolean score. The score type is {score.score_type}.")
        return bool(score.score_value)
