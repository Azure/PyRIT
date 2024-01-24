# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from uuid import uuid4

from pyrit.interfaces import ChatSupport
from pyrit.memory import FileMemory, MemoryInterface
from pyrit.models import ChatMessage, PromptTemplate

RED_TEAM_CHATBOT_ROLE_PREFIX = "airt-bot-uuid"
RED_TEAM_CONVERSATION_END_TOKEN = "<|done|>"


class PromptVariationBot:
    _conversation_memory: MemoryInterface

    def __init__(
        self,
        *,
        conversation_objective: str,
        chat_engine: ChatSupport,
        attack_strategy: PromptTemplate,
        attack_strategy_kwargs: dict[str, str] = None,
        memory: MemoryInterface = None,
        session_id: str = None,
        memory_labels: list[str] = ["red-teaming-bot"],
    ) -> None:
        """Given a prompt

        Args:
            conversation_objective: The objective of the conversation
            base args: Arguments for AzureOpenAIChat
            attack_strategy: The attack strategy to follow by the bot. This can be used to guide the bot to achieve
                the conversation objective in a more direct and structured way. It is a string that can be written in
                a single sentence or paragraph. If not provided, the bot will use an empty string and it will try to
                achieve the conversation objective by itself.
            attack_strategy_kwargs: The attack strategy parameters to use to fill the attack strategy template.
            memory: The memory to use to store the chat messages. If not provided, a FileMemory will be used.
            session_id: The session ID to use for the bot. If not provided, a random UUID will be used.
            memory_labels: The labels to use for the memory. This is useful to identify the bot messages in the memory.
        """
        self._chat_engine = chat_engine
        self._attack_strategy = attack_strategy
        self._global_memory_labels = memory_labels

        # Form the system prompt
        kwargs_to_apply = attack_strategy_kwargs if attack_strategy_kwargs else {}
        kwargs_to_apply["conversation_objective"] = conversation_objective
        self._system_prompt = self._attack_strategy.apply_custom_metaprompt_parameters(**kwargs_to_apply)

        if not memory:
            self._conversation_memory = FileMemory()
        else:
            self._conversation_memory = memory

        self.session_id = session_id if session_id else str(uuid4())

    def __str__(self):
        return f"Red Team bot ID {self.session_id}"

    def get_session_chat_messages(self) -> list[ChatMessage]:
        return self._conversation_memory.get_chat_messages_with_session_id(session_id=self.session_id)

    def complete_chat_user(self, message: str, labels: list[str] = []) -> str:
        message_list: list[ChatMessage] = []
        if not self.get_session_chat_messages():
            # If there are no messages, then this is the first message of the conversation
            message_list.append(ChatMessage(role="system", content=self._system_prompt))

        message_list.append(ChatMessage(role="user", content=message))

        response_msg = self._chat_engine.complete_chat(messages=message_list)
        message_list.append(ChatMessage(role="assistant", content=response_msg))

        self._conversation_memory.add_chat_messages_to_memory(
            conversations=message_list,
            session=self.session_id,
            labels=self._global_memory_labels + labels,
        )

        return response_msg

    def is_conversation_complete(self) -> bool:
        """
        Returns True if the conversation is complete, False otherwise.
        """
        current_messages = self.get_session_chat_messages()

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
