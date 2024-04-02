# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Optional, Union

from pyrit.memory import MemoryInterface
from pyrit.models import AttackStrategy, ChatMessage
from pyrit.orchestrator.red_teaming_orchestrator import RedTeamingOrchestrator
from pyrit.prompt_target import PromptTarget, PromptChatTarget
from pyrit.prompt_converter import PromptConverter

RED_TEAM_CONVERSATION_END_TOKEN = "<|done|>"


class EndTokenRedTeamingOrchestrator(RedTeamingOrchestrator):
    def __init__(
        self,
        *,
        attack_strategy: Union[str, AttackStrategy],
        prompt_target: PromptTarget,
        red_teaming_chat: PromptChatTarget,
        initial_red_teaming_prompt: str = "Begin Conversation",
        end_token: Optional[str] = RED_TEAM_CONVERSATION_END_TOKEN,
        prompt_converters: Optional[list[PromptConverter]] = None,
        memory: Optional[MemoryInterface] = None,
        memory_labels: dict[str, str] = None,
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
            end_token: The token that indicates the end of the conversation.
                If not provided, the default token <|done|> is used.
            prompt_converters: The prompt converters to use to convert the prompts before sending them to the prompt
                target. The converters are not applied on messages to the red teaming target.
            memory: The memory to use to store the chat messages. If not provided, a DuckDBMemory will be used.
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
        self._end_token = end_token
        if end_token not in self._attack_strategy:
            raise ValueError(
                f"Attack strategy must have a way to detect end of conversation and include {end_token} token."
            )

    def is_conversation_complete(self, messages: list[ChatMessage], *, red_teaming_chat_role) -> bool:
        """
        Returns True if the conversation is complete, False otherwise.
        This function checks for the presence of the conversation end token <|done|>
        in the last message by the red teaming chat bot.
        """
        if not messages:
            # If there are no messages, then the conversation is not complete
            return False
        if messages[-1].role != red_teaming_chat_role:
            # If the last message is not sent by the red teaming chat bot then the conversation is not yet complete
            return False
        if self._end_token in messages[-1].content:
            # If the last message contains the conversation end token, then the conversation is complete
            return True
        return False
