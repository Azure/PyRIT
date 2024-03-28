# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import abc
import logging
from typing import Optional, Union
from uuid import uuid4
from pyrit.interfaces import ChatSupport

from pyrit.memory import MemoryInterface
from pyrit.models import AttackStrategy, ChatMessage
from pyrit.orchestrator import Orchestrator
from pyrit.prompt_normalizer import Prompt, PromptNormalizer
from pyrit.prompt_target import PromptTarget
from pyrit.prompt_converter import PromptConverter

logger = logging.getLogger(__name__)

MESSAGE_COUNT_THRESHOLD_TO_INCLUDE_SYSTEM_MESSAGES = 3
MESSAGE_COUNT_WITH_SYSTEM_MESSAGE = 3
MESSAGE_COUNT_DEFAULT = 2


class CompletionState:
    def __init__(self, is_complete: bool):
        self.is_complete = is_complete


class RedTeamingOrchestrator(Orchestrator):
    _memory: MemoryInterface

    def __init__(
        self,
        *,
        attack_strategy: Union[str, AttackStrategy],
        prompt_target: PromptTarget,
        red_teaming_chat: ChatSupport,
        initial_red_teaming_prompt: str = "Begin Conversation",
        prompt_converters: Optional[list[PromptConverter]] = None,
        memory: Optional[MemoryInterface] = None,
        memory_labels: dict[str:str] = None,
        verbose: bool = False,
    ) -> None:
        """Creates an orchestrator to manage conversations between a red teaming target and a prompt target.
           This is an abstract class and needs to be inherited.

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
            prompt_converters: The prompt converters to use to convert the prompts before sending them to the prompt
                target. The converters are not applied on messages to the red teaming target.
            memory: The memory to use to store the chat messages. If not provided, a DuckDBMemory will be used.
            memory_labels: The labels to use for the memory. This is useful to identify the bot messages in the memory.
            verbose: Whether to print debug information.
        """

        super().__init__(
            prompt_converters=prompt_converters, memory=memory, memory_labels=memory_labels, verbose=verbose
        )

        self._prompt_target = prompt_target

        self._prompt_normalizer = PromptNormalizer(memory=self._memory)
        self._prompt_target._memory = self._memory
        self._prompt_target_conversation_id = str(uuid4())
        self._red_teaming_chat_conversation_id = str(uuid4())
        self._red_teaming_chat = red_teaming_chat
        self._attack_strategy = str(attack_strategy)
        self._initial_red_teaming_prompt = initial_red_teaming_prompt

    @property
    def requires_one_to_one_converters(self) -> bool:
        return True

    def get_memory(self):
        return self._memory.get_prompt_entries_with_normalizer_id(normalizer_id=self._prompt_normalizer.id)

    @abc.abstractmethod
    def is_conversation_complete(self, messages: list[ChatMessage], *, red_teaming_chat_role: str) -> bool:
        """Returns True if the conversation is complete, False otherwise."""
        pass

    def apply_attack_strategy_until_completion(self, max_turns: int = 5):
        """
        Applies the attack strategy until the conversation is complete or the maximum number of turns is reached.
        """
        turn = 1
        success = False
        completion_state = CompletionState(is_complete=False)
        overall_response = None
        while turn <= max_turns:
            logger.info(f"Applying the attack strategy for turn {turn}.")
            response = self.send_prompt(completion_state=completion_state)
            # If the conversation is complete without a target response in the current iteration
            # then the overall response is the last iteration's response.
            overall_response = response if response else overall_response
            if completion_state.is_complete:
                success = True
                logger.info(
                    "The red teaming orchestrator has completed the conversation and achieved the objective.",
                )
                break
            turn += 1

        if not success:
            logger.info(
                "The red teaming orchestrator has not achieved the objective after the maximum "
                f"number of turns ({max_turns}).",
            )

        return overall_response

    def send_prompt(self, *, prompt: Optional[str] = None, completion_state: CompletionState = None):
        """
        Either sends a user-provided prompt or generates a prompt to send to the prompt target.

        Args:
            prompt: The prompt to send to the target.
                If no prompt is specified the orchestrator contacts the red teaming target
                to generate a prompt and forwards it to the prompt target.
                This can only be specified for the first iteration at this point.
            completion_state: The completion state of the conversation.
                If the conversation is complete, the completion state is updated to True.
                It is optional and not tracked if no object is passed.
        """
        target_messages = self._memory.get_chat_messages_with_conversation_id(
            conversation_id=self._prompt_target_conversation_id
        )
        if prompt:
            if target_messages:
                raise ValueError("The prompt argument can only be provided on the first iteration. ")
        else:
            # If no prompt is provided, then contact the red teaming target to generate one.
            # The prompt for the red teaming LLM needs to include the latest message from the prompt target.
            # A special case is the very first message, which means there are no prior messages.
            logger.info(
                "No prompt for prompt target provided. "
                "Generating a prompt for the prompt target using the red teaming LLM."
            )

            assistant_responses = [m for m in target_messages if m.role == "assistant"]
            if len(assistant_responses) > 0:
                prompt_text = assistant_responses[-1].content
            else:  # If no assistant responses, then it's the first message
                logger.info(f"Using the specified initial red teaming prompt: {self._initial_red_teaming_prompt}")
                prompt_text = self._initial_red_teaming_prompt

            logger.info(f'Sending the following prompt to the red teaming prompt target "{prompt_text}"')
            messages = self._memory.get_chat_messages_with_conversation_id(
                conversation_id=self._red_teaming_chat_conversation_id
            )
            if not messages:
                messages.append(ChatMessage(role="system", content=self._attack_strategy))
            messages.append(ChatMessage(role="user", content=prompt_text))
            prompt = self._red_teaming_chat.complete_chat(messages=messages)
            if not prompt:
                raise ValueError("The red teaming chat returned an empty prompt. Run with verbose=True to debug.")
            messages.append(ChatMessage(role="assistant", content=prompt))
            # Determine the number of messages to add to memory based on if we included the system message
            memory_messages = MESSAGE_COUNT_DEFAULT
            if len(messages) <= MESSAGE_COUNT_THRESHOLD_TO_INCLUDE_SYSTEM_MESSAGES:
                memory_messages = MESSAGE_COUNT_WITH_SYSTEM_MESSAGE
            conversations_to_be_added_to_memory = messages[-memory_messages:]

            self._memory.add_chat_messages_to_memory(
                conversations=conversations_to_be_added_to_memory,
                conversation_id=self._red_teaming_chat_conversation_id,
                labels=self._global_memory_labels,
            )

        if completion_state and self.is_conversation_complete(messages, red_teaming_chat_role="assistant"):
            completion_state.is_complete = True
            return

        logger.info(
            "Sending the following prompt to the prompt target (after applying prompt "
            f'converter operations) "{prompt}"',
        )
        target_prompt_obj = Prompt(
            prompt_target=self._prompt_target,
            prompt_converters=self._prompt_converters,
            prompt_text=prompt,
            conversation_id=self._prompt_target_conversation_id,
        )
        response = self._prompt_normalizer.send_prompt(prompt=target_prompt_obj)[0]
        logger.info(f'Received the following response from the prompt target "{response}"')
        if completion_state:
            target_messages.append(ChatMessage(role="user", content=prompt))
            target_messages.append(ChatMessage(role="assistant", content=response))
            completion_state.is_complete = self.is_conversation_complete(target_messages, red_teaming_chat_role="user")
        return response
