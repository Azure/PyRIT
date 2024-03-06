# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import abc
import logging
import pathlib
from typing import Optional, Union, Callable
from uuid import uuid4

from pyrit.memory import FileMemory, MemoryInterface
from pyrit.models import AttackStrategy, ChatMessage
from pyrit.common.path import DATASETS_PATH
from pyrit.prompt_normalizer import Prompt, PromptNormalizer
from pyrit.prompt_target import PromptTarget
from pyrit.prompt_converter import PromptConverter, NoOpConverter


logger = logging.getLogger(__name__)

RED_TEAM_CHATBOT_ROLE_PREFIX = "airt-bot-uuid"
RED_TEAM_CONVERSATION_END_TOKEN = "<|done|>"


def _default_is_conversation_complete(messages: list[ChatMessage]) -> bool:
    """
    Returns True if the conversation is complete, False otherwise.
    This function checks for the presence of the conversation end token <|done|>.
    """
    if not messages:
        # If there are no messages, then the conversation is not complete
        return False
    if messages[-1].role == "system":
        # If the last message is a system message, then the conversation is not yet complete
        return False
    if RED_TEAM_CONVERSATION_END_TOKEN in messages[-1].content:
        # If the last message contains the conversation end token, then the conversation is complete
        return True
    return False
    

class RedTeamingOrchestrator:
    _memory: MemoryInterface

    def __init__(
        self,
        *,
        attack_strategy: Union[str, AttackStrategy],
        prompt_target: PromptTarget,
        red_teaming_target: PromptTarget,
        initial_red_teaming_prompt: str,
        is_conversation_complete: Callable[[list[ChatMessage]], bool] = _default_is_conversation_complete,
        prompt_converter: Optional[PromptConverter] = None,
        memory: Optional[MemoryInterface] = None,
        memory_labels: list[str] = ["red-teaming-orchestrator"],
        verbose: bool = False,
    ) -> None:
        """Creates an orchestrator to manage conversations between

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
        self._red_teaming_target_conversation_id = str(uuid4())
        self._red_teaming_target = red_teaming_target
        self._attack_strategy = str(attack_strategy)
        self._initial_red_teaming_prompt = initial_red_teaming_prompt
        self._global_memory_labels = memory_labels
        self._is_conversation_complete = is_conversation_complete

        # Form the system prompt
        self._red_teaming_target.set_system_prompt(self._attack_strategy, self._red_teaming_target_conversation_id, self._prompt_normalizer.id)

    def get_memory(self):
        return self._memory.get_memories_with_normalizer_id(normalizer_id=self._prompt_normalizer.id)        

    def apply_attack_strategy_until_completion(self, max_turns: int = 10):
        """
        Applies the attack strategy until the conversation is complete or the maximum number of turns is reached.
        """
        turn = 0
        while turn < max_turns and \
                not self._is_conversation_complete(
                    self._memory.get_chat_messages_with_conversation_id(
                        conversation_id=self._red_teaming_target_conversation_id)
                ) and turn < max_turns:
            self.send_prompt()
            turn += 1

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
            messages = self._memory.get_chat_messages_with_conversation_id(conversation_id=self._prompt_target_conversation_id)
            assistant_responses = [m for m in messages if m.role == "assistant"]
            if len(assistant_responses) > 0:
                prompt_text = messages[-1].content
            else: # If no assistant responses, then it's the first message
                logger.log(logging.INFO, f"Using the specified initial red teaming prompt.")
                prompt_text = self._initial_red_teaming_prompt

            logger.log(logging.INFO, f"Sending the following prompt to the red teaming prompt target \"{prompt_text}\"")
            red_teaming_prompt_obj = Prompt(
                prompt_target=self._red_teaming_target,
                prompt_converter=NoOpConverter(),
                prompt_text=prompt_text,
                conversation_id=self._red_teaming_target_conversation_id,
            )

            prompt = self._prompt_normalizer.send_prompt(prompt=red_teaming_prompt_obj)

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
