# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import pathlib
from uuid import uuid4

from pyrit.interfaces import ChatSupport
from pyrit.memory import FileMemory, MemoryInterface
from pyrit.models import ChatMessage, PromptTemplate
from pyrit.common.path import HOME_PATH

RED_TEAM_CHATBOT_ROLE_PREFIX = "airt-bot-uuid"
RED_TEAM_CONVERSATION_END_TOKEN = "<|done|>"


class RedTeamingBot:
    _conversation_memory: MemoryInterface

    def __init__(
        self,
        *,
        conversation_objective: str,
        chat_engine: ChatSupport,
        attack_strategy: PromptTemplate = None,
        attack_strategy_kwargs: dict[str, str] = None,
        memory: MemoryInterface = None,
        conversation_id: str = None,
        memory_labels: list[str] = ["red-teaming-bot"],
    ) -> None:
        """Created an adversarial chatbot that can be used to test the security of a chatbot.

        Args:
            conversation_objective: The objective of the conversation
            base args: Arguments for AzureOpenAIChat
            attack_strategy: The attack strategy to follow by the bot. This can be used to guide the bot to achieve
                the conversation objective in a more direct and structured way. It is a string that can be written in
                a single sentence or paragraph. If not provided, the bot will use red_team_chatbot_with_objective.
            attack_strategy_kwargs: The attack strategy parameters to use to fill the attack strategy template.
            memory: The memory to use to store the chat messages. If not provided, a FileMemory will be used.
            conversation_id: The session ID to use for the bot. If not provided, a random UUID will be used.
            memory_labels: The labels to use for the memory. This is useful to identify the bot messages in the memory.
        """
        self._chat_engine = chat_engine
        self._attack_strategy = (
            attack_strategy
            if attack_strategy
            else PromptTemplate.from_yaml_file(
                pathlib.Path(HOME_PATH)
                / "datasets"
                / "attack_strategies"
                / "multi_turn_chat"
                / "red_team_chatbot_with_objective.yaml"
            )
        )

        if RED_TEAM_CONVERSATION_END_TOKEN not in self._attack_strategy.template:
            raise ValueError(
                f"Attack strategy must have a way to detect end of conversation and include \
                {RED_TEAM_CONVERSATION_END_TOKEN} token."
            )

        self._global_memory_labels = memory_labels

        # Form the system prompt
        kwargs_to_apply = attack_strategy_kwargs if attack_strategy_kwargs else {}
        kwargs_to_apply["conversation_objective"] = conversation_objective
        self._system_prompt = self._attack_strategy.apply_custom_metaprompt_parameters(**kwargs_to_apply)

        self._conversation_memory = memory if memory else FileMemory()
        self.conversation_id = conversation_id if conversation_id else str(uuid4())

    def __str__(self):
        return f"Red Team bot ID {self.conversation_id}"

    def get_conversation_chat_messages(self) -> list[ChatMessage]:
        return self._conversation_memory.get_chat_messages_with_conversation_id(conversation_id=self.conversation_id)

    def complete_chat_user(self, message: str, labels: list[str] = []) -> str:
        chat_entries = self.get_conversation_chat_messages()

        if not chat_entries:
            # If there are no messages, then this is the first message of the conversation
            chat_entries.append(ChatMessage(role="system", content=self._system_prompt))

        chat_entries.append(ChatMessage(role="user", content=message))

        response_msg = self._chat_engine.complete_chat(messages=chat_entries)

        chat_entries.append(ChatMessage(role="assistant", content=response_msg))

        # Determine the number of messages to add to memory based on if we included the system message
        memory_messages = 3 if len(chat_entries) <= 3 else 2

        # Add the last two or three new messages to memory
        self._conversation_memory.add_chat_messages_to_memory(
            conversations=chat_entries[-memory_messages:],
            conversation_id=self.conversation_id,
            labels=self._global_memory_labels + labels,
        )

        return response_msg

    def is_conversation_complete(self) -> bool:
        """
        Returns True if the conversation is complete, False otherwise.
        """
        current_messages = self.get_conversation_chat_messages()

        if not current_messages or len(current_messages) == 0:
            # If there are no messages, then the conversation is not complete
            return False
        if current_messages[-1].role == "system":
            # If the last message is a system message, then the conversation is complete
            return False

        if RED_TEAM_CONVERSATION_END_TOKEN in current_messages[-1].content:
            # If the last message contains the conversation end token, then the conversation is complete
            return True
        return False
