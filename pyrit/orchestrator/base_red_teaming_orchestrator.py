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
    

class BaseRedTeamingOrchestrator:
    _memory: MemoryInterface

    def __init__(
        self,
        *,
        attack_strategy: Union[str, AttackStrategy],
        prompt_target: PromptTarget,
        red_teaming_chat: ChatSupport,
        initial_red_teaming_prompt: str,
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
            red_teaming_chat: The endpoint that creates prompts that are sent to the prompt target.
            initial_red_teaming_prompt: The initial prompt to send to the red teaming target.
                The attack_strategy only provides the strategy, but not the starting point of the conversation.
                The initial_red_teaming_prompt is used to start the conversation with the red teaming target.
            prompt_converter: The prompt converter to use to convert the prompts before sending them to the prompt
                target. The converter is not applied in messages to the red teaming target.
            memory: The memory to use to store the chat messages. If not provided, a FileMemory will be used.
            memory_labels: The labels to use for the memory. This is useful to identify the bot messages in the memory.
            verbose: Whether to print debug information.
        """
        self._verbose = verbose
        if self._verbose:
            logging.basicConfig(level=logging.INFO)
        self._prompt_target = prompt_target
        self._prompt_converter = prompt_converter if prompt_converter else NoOpConverter()
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
    def is_conversation_complete(self, messages: list[ChatMessage]) -> bool:
        """Returns True if the conversation is complete, False otherwise."""
        pass

    def apply_attack_strategy_until_completion(self, max_turns: int = 10):
        """
        Applies the attack strategy until the conversation is complete or the maximum number of turns is reached.
        """
        turn = 1
        success = False
        while turn <= max_turns:
            logger.log(logging.INFO, f"Applying the attack strategy for turn {turn}.")
            self.send_prompt()
            if self.is_conversation_complete(
                self._memory.get_chat_messages_with_conversation_id(
                    conversation_id=self._red_teaming_chat_conversation_id)
            ):
                success = True
                logger.log(logging.INFO, "The red teaming orchestrator has completed the conversation and achieved the objective.")
                break
            turn += 1

        if not success:
            logger.log(logging.INFO, f"The red teaming orchestrator has not achieved the objective after the maximum number of turns ({max_turns}).")

    def send_prompt(self, prompt: Optional[str] = None):
        """
        Either sends a user-provided prompt or generates a prompt to send to the prompt target.

        Args:
            prompt: The prompt to send to the target.
                If no prompt is specified the orchestrator contacts the red teaming target
                to generate a prompt and forwards it to the prompt target.
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
                messages=messages[-memory_messages:],
                conversation_id=self._red_teaming_chat_conversation_id,
                labels=self._global_memory_labels,
            )

        logger.log(logging.INFO, f"Sending the following prompt to the prompt target (after applying prompt converter operations) \"{prompt}\"")
        target_prompt_obj = Prompt(
            prompt_target=self._prompt_target,
            prompt_converter=self._prompt_converter,
            prompt_text=prompt,
            conversation_id=self._prompt_target_conversation_id,
        )
        response = self._prompt_normalizer.send_prompt(prompt=target_prompt_obj)
        logger.log(logging.INFO, f"Received the following response from the prompt target \"{response}\"")
        return response
