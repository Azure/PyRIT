# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import abc
import logging
from typing import Optional, Union
from uuid import uuid4
from pyrit.interfaces import ChatSupport

from pyrit.memory import FileMemory, MemoryInterface
from pyrit.models import AttackStrategy, ChatMessage
from pyrit.prompt_normalizer import Prompt, PromptNormalizer
from pyrit.prompt_target import PromptTarget
from pyrit.prompt_converter import PromptConverter, NoOpConverter


logger = logging.getLogger(__name__)


class CompletionState:
    def __init__(self, is_complete: bool):
        self.is_complete = is_complete


class BaseRedTeamingOrchestrator:
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
        memory_labels: list[str] = ["red-teaming-orchestrator"],
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
            prompt_converters: The prompt converters to use to convert the prompts before sending them to the prompt
                target. The converters are not applied on messages to the red teaming target.
            memory: The memory to use to store the chat messages. If not provided, a FileMemory will be used.
            memory_labels: The labels to use for the memory. This is useful to identify the bot messages in the memory.
            verbose: Whether to print debug information.
        """
        self._verbose = verbose
        if self._verbose:
            logging.basicConfig(level=logging.INFO)
        self._prompt_target = prompt_target
        self._prompt_converters = prompt_converters if prompt_converters else [NoOpConverter()]

        # Ensure that all converters return exactly one prompt.
        # Otherwise, there will be more than 1 conversation to manage.
        one_to_many_converters = []
        for converter in self._prompt_converters:
            if not converter.is_one_to_one_converter():
                one_to_many_converters.append(str(converter))
        if one_to_many_converters:
            one_to_many_converters_str = ", ".join(one_to_many_converters)
            raise ValueError(
                f"The following converters create more than one prompt: {one_to_many_converters_str}")

        self._memory = memory if memory else FileMemory()
        self._prompt_normalizer = PromptNormalizer(memory=self._memory)
        self._prompt_target.memory = self._memory
        self._prompt_target_conversation_id = str(uuid4())
        self._red_teaming_chat_conversation_id = str(uuid4())
        self._red_teaming_chat = red_teaming_chat
        self._attack_strategy = str(attack_strategy)
        self._initial_red_teaming_prompt = initial_red_teaming_prompt
        self._global_memory_labels = memory_labels

    def get_memory(self):
        return self._memory.get_memories_with_normalizer_id(normalizer_id=self._prompt_normalizer.id)        

    @abc.abstractmethod
    def is_conversation_complete(self, messages: list[ChatMessage], *, red_teaming_chat_role) -> bool:
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
            logger.log(logging.INFO, f"Applying the attack strategy for turn {turn}.")
            response = self.send_prompt(completion_state=completion_state)
            # If the conversation is complete without a target response in the current iteration
            # then the overall response is the last iteration's response.
            overall_response = response if response else overall_response
            if completion_state.is_complete:
                success = True
                logger.log(logging.INFO, "The red teaming orchestrator has completed the conversation and achieved the objective.")
                break
            turn += 1

        if not success:
            logger.log(logging.INFO, f"The red teaming orchestrator has not achieved the objective after the maximum number of turns ({max_turns}).")
        
        return overall_response

    def send_prompt(self, *, prompt: Optional[str] = None, completion_state: CompletionState = None):
        """
        Either sends a user-provided prompt or generates a prompt to send to the prompt target.

        Args:
            prompt: The prompt to send to the target.
                If no prompt is specified the orchestrator contacts the red teaming target
                to generate a prompt and forwards it to the prompt target.
            completion_state: The completion state of the conversation.
                If the conversation is complete, the completion state is updated to True.
                It is optional and not tracked if no object is passed.
        """
        if not prompt:
            # If no prompt is provided, then contact the red teaming target to generate one.
            # The prompt for the red teaming LLM needs to include the latest message from the prompt target.
            # A special case is the very first message, which means there are no prior messages.
            logger.log(logging.INFO, "No prompt for prompt target provided.")
            target_messages = self._memory.get_chat_messages_with_conversation_id(conversation_id=self._prompt_target_conversation_id)
            assistant_responses = [m for m in target_messages if m.role == "assistant"]
            if len(assistant_responses) > 0:
                prompt_text = assistant_responses[-1].content
            else: # If no assistant responses, then it's the first message
                logger.log(logging.INFO, f"Using the specified initial red teaming prompt.")
                prompt_text = self._initial_red_teaming_prompt

            logger.log(logging.INFO, f"Sending the following prompt to the red teaming prompt target \"{prompt_text}\"")
            messages = self._memory.get_chat_messages_with_conversation_id(conversation_id=self._red_teaming_chat_conversation_id)
            if not messages:
                messages.append(ChatMessage(role="system", content=self._attack_strategy))
            messages.append(ChatMessage(role="user", content=prompt_text))
            prompt = self._red_teaming_chat.complete_chat(messages=messages)
            messages.append(ChatMessage(role="assistant", content=prompt))
            # Determine the number of messages to add to memory based on if we included the system message
            memory_messages = 3 if len(messages) <= 3 else 2
            self._memory.add_chat_messages_to_memory(
                conversations=messages[-memory_messages:],
                conversation_id=self._red_teaming_chat_conversation_id,
                labels=self._global_memory_labels,
            )

        if completion_state and self.is_conversation_complete(
            messages, red_teaming_chat_role="assistant"
        ):
            completion_state.is_complete = True
            return

        logger.log(
            logging.INFO,
            "Sending the following prompt to the prompt target (after applying prompt "
            f"converter operations) \"{prompt}\"")
        target_prompt_obj = Prompt(
            prompt_target=self._prompt_target,
            prompt_converters=self._prompt_converters,
            prompt_text=prompt,
            conversation_id=self._prompt_target_conversation_id,
        )
        response = self._prompt_normalizer.send_prompt(prompt=target_prompt_obj)[0]
        logger.log(logging.INFO, f"Received the following response from the prompt target \"{response}\"")
        if completion_state and self.is_conversation_complete(
            target_messages + [
                ChatMessage(role="user", content=prompt),
                ChatMessage(role="assistant", content=response),
            ],
            red_teaming_chat_role="user"
        ):
            completion_state.is_complete = True
        return response
